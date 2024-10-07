import aioboto3
from boto3.dynamodb.conditions import Key  
from typing import Any, Dict, AsyncIterator, Optional, Sequence, Tuple
from langchain_core.messages import BaseMessage
import langchain_core.messages as langchain_messages
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
import time
import json

from langchain_core.runnables import RunnableConfig

from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
)

class JsonPlusSerializer(JsonPlusSerializer):
    def dumps_typed(self, obj: Any) -> Tuple[str, Any]:
        def default(o):
            if isinstance(o, BaseMessage):
                return {
                    '__type__': o.__class__.__name__,
                    'data': o.model_dump(),
                }
            raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

        json_str = json.dumps(obj, default=default)
        return 'json', json_str

    def loads_typed(self, data: Tuple[str, Any]) -> Any:
        type_, json_str = data

        def object_hook(dct):
            if '__type__' in dct:
                type_name = dct['__type__']
                data = dct['data']
                cls = getattr(langchain_messages, type_name, None)
                if cls and issubclass(cls, BaseMessage):
                    return cls.model_construct(**data)
                else:
                    raise ValueError(f'Unknown type: {type_name}')
            return dct

        obj = json.loads(json_str, object_hook=object_hook)
        return obj

class AsyncDynamoDBSaver(BaseCheckpointSaver):
    """An asynchronous checkpoint saver that stores checkpoints in DynamoDB using JSON-compatible formats."""

    WIDTH = 20  # Width for zero-padding timestamps

    def __init__(
        self,
        table_name: str,
        writes_table_name: str,
        region_name: str = 'us-west-2',
        endpoint_url: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.serde = JsonPlusSerializer()
        self.table_name = table_name
        self.writes_table_name = writes_table_name
        self.region_name = region_name
        self.endpoint_url = endpoint_url
        self.session = aioboto3.Session() 

    async def __aenter__(self):
        # Use the session to get the DynamoDB resource as an async context manager
        self.dynamodb_context = self.session.resource(
            'dynamodb',
            region_name=self.region_name,
            endpoint_url=self.endpoint_url,
        )
        # Enter the asynchronous context manager
        self.dynamodb = await self.dynamodb_context.__aenter__()
        # Await the Table coroutines to get the actual table objects
        self.table = await self.dynamodb.Table(self.table_name)
        self.writes_table = await self.dynamodb.Table(self.writes_table_name)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Exit the asynchronous context manager
        await self.dynamodb_context.__aexit__(exc_type, exc_val, exc_tb)

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        """Fetch a checkpoint tuple using a given configuration asynchronously."""
        
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = get_checkpoint_id(config)

        if checkpoint_id:
            # Need to scan for item with matching checkpoint_id
            response = await self.table.query(
                KeyConditionExpression=Key('thread_id').eq(thread_id) & Key('sort_key').begins_with(f'{checkpoint_ns}#'),
                FilterExpression=Key('checkpoint_id').eq(checkpoint_id)
            )
            items = response.get('Items', [])
            if not items:
                return None
            item = items[0]
        else:
            # Fetch the latest checkpoint for the thread_id and checkpoint_ns
            response = await self.table.query(
                KeyConditionExpression=Key('thread_id').eq(thread_id) & Key('sort_key').begins_with(f'{checkpoint_ns}#'),
                ScanIndexForward=False,  # Descending order
                Limit=1
            )
            items = response.get('Items', [])
            if not items:
                return None
            item = items[0]

        # Reconstruct the checkpoint tuple from the item
        sort_key_parts = item['sort_key'].split('#')
        checkpoint_ns = sort_key_parts[0]
        checkpoint_id = item['checkpoint_id']

        config_values = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
        checkpoint_type = item['type']

        # Retrieve the checkpoint data directly
        checkpoint_data = item['checkpoint']

        # Deserialize checkpoint data
        checkpoint = self.serde.loads_typed((checkpoint_type, checkpoint_data))

        # Get pending writes from "checkpoint_writes" table
        write_sort_key_prefix = f'{checkpoint_ns}#{checkpoint_id}#'
        pending_writes = []

        # Query writes_table
        response = await self.writes_table.query(
            KeyConditionExpression=Key('thread_id').eq(thread_id) & Key('sort_key').begins_with(write_sort_key_prefix),
            ScanIndexForward=True
        )
        write_items = response.get('Items', [])

        for write_item in write_items:
            task_id = write_item['task_id']
            channel = write_item['channel']
            value_type = write_item['type']

            # Retrieve the value data directly
            value_data = write_item['value']

            value = self.serde.loads_typed((value_type, value_data))
            pending_writes.append((task_id, channel, value))

        # Retrieve metadata directly
        metadata = item['metadata']
        metadata = self.serde.loads(metadata)
        # Get parent config if any
        parent_checkpoint_id = item.get('parent_checkpoint_id')
        if parent_checkpoint_id:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": parent_checkpoint_id,
                }
            }
        else:
            parent_config = None

        return CheckpointTuple(
            {"configurable": config_values},
            checkpoint,
            metadata,
            parent_config,
            pending_writes,
        )

    async def alist(
        self,
        config: Optional[RunnableConfig],
        *,
        filter: Optional[Dict[str, Any]] = None,
        before: Optional[RunnableConfig] = None,
        limit: Optional[int] = None,
    ) -> AsyncIterator[CheckpointTuple]:
        """List checkpoints that match a given configuration and filter criteria asynchronously."""
        
        if config is None:
            raise ValueError("config must be provided for listing checkpoints in DynamoDB")

        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_ns_prefix = f'{checkpoint_ns}#'

        query_kwargs = {
            'KeyConditionExpression': Key('thread_id').eq(thread_id) & Key('sort_key').begins_with(checkpoint_ns_prefix),
            'ScanIndexForward': False  # Descending order
        }

        if limit is not None:
            query_kwargs['Limit'] = limit

        response = await self.table.query(**query_kwargs)
        items = response.get('Items', [])

        for item in items:
            sort_key_parts = item['sort_key'].split('#')
            checkpoint_ns = sort_key_parts[0]
            checkpoint_id = item['checkpoint_id']
            checkpoint_type = item['type']

            # Retrieve the checkpoint data directly
            checkpoint_data = item['checkpoint']

            checkpoint = self.serde.loads_typed((checkpoint_type, checkpoint_data))

            # Retrieve metadata directly
            metadata = item['metadata']
            metadata = self.serde.loads(metadata)

            parent_checkpoint_id = item.get('parent_checkpoint_id')
            if parent_checkpoint_id:
                parent_config = {
                    "configurable": {
                        "thread_id": item['thread_id'],
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_checkpoint_id,
                    }
                }
            else:
                parent_config = None

            yield CheckpointTuple(
                {
                    "configurable": {
                        "thread_id": item['thread_id'],
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": checkpoint_id,
                    }
                },
                checkpoint,
                metadata,
                parent_config,
            )

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        """Store a checkpoint with its configuration and metadata asynchronously."""
        
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = checkpoint["id"]
        checkpoint_created_at = int(time.time() * 1000)
        sort_key = f'{checkpoint_ns}#{checkpoint_created_at:0{self.WIDTH}d}'

        type_, checkpoint_data = self.serde.dumps_typed(checkpoint)

        parent_checkpoint_id = config["configurable"].get("checkpoint_id")

        item = {
            'thread_id': thread_id,
            'sort_key': sort_key,
            'checkpoint_id': checkpoint_id,
            'parent_checkpoint_id': parent_checkpoint_id,
            'type': type_,
            'checkpoint': checkpoint_data,
            'metadata': self.serde.dumps_typed(metadata)[1],
        }

        await self.table.put_item(Item=item)

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        """Store intermediate writes linked to a checkpoint (i.e., pending writes) asynchronously."""
        
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"]["checkpoint_ns"]
        checkpoint_id = config["configurable"]["checkpoint_id"]

        async with self.writes_table.batch_writer() as batch:
            for idx, (channel, value) in enumerate(writes):
                idx_str = f'{idx:0{self.WIDTH}d}'
                sort_key = f'{checkpoint_ns}#{checkpoint_id}#{task_id}#{idx_str}'
                type_, value_data = self.serde.dumps_typed(value)

                item = {
                    'thread_id': thread_id,
                    'sort_key': sort_key,
                    'task_id': task_id,
                    'idx': idx,
                    'channel': channel,
                    'type': type_,
                    'value': value_data,
                }
                await batch.put_item(Item=item)
