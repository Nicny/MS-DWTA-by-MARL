import os
from runner import Runner
from mozi_ai_sdk.MS_DWTA_open_source.env import AirDefense as Environment
from mozi_ai_sdk.MS_DWTA_open_source.marl.arguments import get_common_args, get_algo_args

os.environ['MOZIPATH'] = 'D:/Mozi/MoziServer/bin'

args = get_common_args()
args = get_algo_args(args)

env = Environment(server_ip=args.server_ip,
                  server_port=args.server_port,
                  agent_key_event_file=None,
                  duration_interval=args.duration_interval,
                  app_mode=args.appmode,
                  synchronous=args.synchronous,
                  simulate_compression=args.simulate_compression,
                  scenario_name=args.scenario_name,
                  platform_mode=args.platform_mode,
                  args=args)
env.start()
runner = Runner(env, args)

if args.evaluate:
    runner.evaluate(0)
else:
    runner.run(1)

