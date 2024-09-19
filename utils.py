import json
import textwrap


def load_config(config_path='config.json'):
    with open(config_path) as json_file:
        return json.load(json_file)

def append_reward_to_env(reward_file, orig_env_file="env_orig.py", output_env_file="env.py"):
    with open(orig_env_file, 'r') as orig_file, open(reward_file, 'r') as reward_file:
        orig_code = orig_file.read()
        reward_code = reward_file.read()

    indented_reward_code = textwrap.indent(reward_code, ' ' * 4)

    with open(output_env_file, 'w') as output_file:
        output_file.write(orig_code)
        output_file.write('\n' + indented_reward_code + '\n')

    return output_env_file
