import os
import dotenv

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)


def save_env_vars():
    dotenv.set_key(dotenv_file, "AGENT_COUNTER", os.environ["AGENT_COUNTER"])


def is_collab():
    return not ('USER' in os.environ and os.environ['USER'] == 'robert')


def next_agent_id():
    counter = int(os.environ.get("AGENT_COUNTER", 0))
    os.environ['AGENT_COUNTER'] = str(counter + 1)
    return counter


def set_agent_id(agent_id):
    os.environ['AGENT_COUNTER'] = str(agent_id)


def get_devices():
    devices = os.environ.get('DEVICES', '').split(',')
    return [d.strip() for d in devices if len(d.strip()) > 0]
