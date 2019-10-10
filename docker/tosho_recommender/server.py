import argparse
from aiohttp import web
import json
from handler import recommend_book

# Configure server
parser = argparse.ArgumentParser(description="aiohttp server example")
parser.add_argument('--port')
args = parser.parse_args()

async def handle_post(request):
   body = await request.json()
   try:
       book1 = body['book1']
       book2 = body['book2']
       book3 = body['book3']
       book4 = body['book4']
   except:
       book1 = u""
       book2 = u""
       book3 = u""
       book4 = u""
   return web.json_response(text=json.dumps(recommend_book([book1, book2, book3, book4]), ensure_ascii = False))
   
async def health_check(request):
   return web.Response()

# Configure server routes
app = web.Application()
app.router.add_post('/', handle_post)
app.router.add_get('/health-check', health_check)
print("app running...")
web.run_app(app, port=int(args.port))
