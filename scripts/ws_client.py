import asyncio
import os

import aiohttp
import json
import numpy as np

import subprocess

HOST = os.getenv('HOST', '172.29.229.220')
# HOST = os.getenv('HOST', '0.0.0.0')
# HOST = os.getenv('HOST', 'KLAB-BUTTER.PC.CS.CMU.EDU')
PORT = int(os.getenv('PORT', 8080))


async def main():
    session = aiohttp.ClientSession()
    URL = f'http://{HOST}:{PORT}/ws_talk'
    async with session.ws_connect(URL) as ws:

        await prompt_and_send(ws)
        async for msg in ws:
            print('Message received from server:', msg.data)
            await prompt_and_send(ws)
            if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                break

    # session = aiohttp.ClientSession()
    # URL = f'http://{HOST}:{PORT}/ws'
    # import time
    # async with session.ws_connect(URL) as ws:
    #     await ws.send_str("get_pose")
    #     async for msg in ws:
    #         t_s = time.time()
    #         json_data = json.loads(msg.data)
    #         print(json_data['pose_mat'][0])

    #         await ws.send_str("get_pose")

    #         if msg.type in (aiohttp.WSMsgType.CLOSED,
    #                         aiohttp.WSMsgType.ERROR):
    #             break

    #         await asyncio.sleep(1/30)

    #         dt = time.time() - t_s
    #         print(1/dt)


async def prompt_and_send(ws):
    new_msg_to_send = input('Type a message to send to the server: ')
    if new_msg_to_send == 'exit':
        print('Exiting!')
        raise SystemExit(0)
    elif new_msg_to_send == "s":
        # subprocess.Popen(["simplescreenrecorder", "--start-recording"])
        pass
    elif new_msg_to_send == "e":
        pass

    await ws.send_str(new_msg_to_send)
    return new_msg_to_send


if __name__ == '__main__':
    print('Type "exit" to quit')
    # loop = asyncio.get_event_loop()
    # loop.run_forever(main())
    asyncio.run(main())