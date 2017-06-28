import pandas as pd
import pickle


if __name__ == '__main__':
    user_app_action = pickle.load(open('../saved_file/user_app_action.pkl','rb'))
    print(user_app_action.head())
    print(min(user_app_action['install_day']))
    print(max(user_app_action['install_day']))