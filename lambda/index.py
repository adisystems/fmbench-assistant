from lambda.lambda import app, generate_route
import json
import asyncio

def handler(event, context):
    try:
        # For Lambda URL invocations, the body is a string that needs to be parsed
        body = json.loads(event.get('body', '{}'))
        
        # Create a request object with the necessary data
        request = {'question': body.get('question'), 'thread_id': body.get('thread_id')}
        
        # Run the async function
        if asyncio._get_running_loop() is not None:
            loop = asyncio._get_running_loop()
        else:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(generate_route(request))
        
        # Return Lambda URL compatible response with CORS headers
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Content-Type': 'application/json'
            },
            'body': json.dumps(result)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({'error': str(e)})
        }