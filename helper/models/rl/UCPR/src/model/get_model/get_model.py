from helper.models.rl.UCPR.src.env.env import BatchKGEnvironment
from helper.models.rl.UCPR.src.model.baseline.baseline import ActorCritic
from helper.models.rl.UCPR.src.model.lstm_base.model_lstm_mf_emb import AC_lstm_mf_dummy
from helper.models.rl.UCPR.parser import parse_args
from helper.models.rl.UCPR.src.model.UCPR import UCPR


args = parse_args()

# ********************* model select *****************************

if args.model == 'lstm': 
    Memory_Model = AC_lstm_mf_dummy
elif args.model == 'UCPR':
    Memory_Model = UCPR
elif args.model == 'baseline':
    Memory_Model = ActorCritic

# ********************* BatchKGEnvironment ************************

KGEnvironment = BatchKGEnvironment 

