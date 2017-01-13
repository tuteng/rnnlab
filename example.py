from src.rnnlab import RNN
from src.rnnlab import gen_user_configs

for user_configs in gen_user_configs():
    myrnn = RNN('lstm', user_configs) # try 'srn', ''irnn', 'scrn'
    myrnn.train()


