from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pickle
import numpy as np
import os


def iplprediction(inningFile,predictionFile):

    df = pd.read_csv(inningFile)
    dft = pd.read_csv(predictionFile)

    df['Wickets']=0
    df["Final_Score"]=0
    df["Current_Score"]=0
    df.replace(np.NaN, int(0), inplace=True)
    for i in range(1,df.shape[0]):
        if df.loc[i, "match_id"] != df.loc[i-1,"match_id"]:
            df.loc[i,"Wickets"]=0
            if df.loc[i, "player_dismissed"] != 0:
                df.loc[i,"Wickets"]=1
        else:
            if df.loc[i, "player_dismissed"] != 0:
                df.loc[i,"Wickets"]=df.loc[i-1,"Wickets"]+1
            else:
                df.loc[i,"Wickets"]=df.loc[i-1,"Wickets"]
    
    columns_to_remove = ["player_dismissed","dismissal_kind","fielder", "wide_runs","bye_runs","legbye_runs", "penalty_runs", "noball_runs","batsman_runs","extra_runs"]
    df.drop(labels=columns_to_remove, axis=1, inplace=True)

    for i in range(1,df.shape[0]):
        if df.loc[i, "match_id"] != df.loc[i-1,"match_id"]:
            df.loc[i,"Final_Score"]= df.loc[i,"total_runs"]
        else:
            df.loc[i,"Final_Score"]=df.loc[i-1,"Final_Score"]+df.loc[i,"total_runs"]

    for i in range(df.shape[0]-2,-1,-1):
        if df.loc[i, "match_id"] != df.loc[i+1,"match_id"]:
            df.loc[i,"Final_Score"]= df.loc[i,"Final_Score"]
        else:
            df.loc[i,"Final_Score"]=df.loc[i+1,"Final_Score"]

    for i in range(1,df.shape[0]):
        if df.loc[i, "match_id"] != df.loc[i-1,"match_id"]:
            df.loc[i,"Current_Score"]= df.loc[i,"total_runs"]
        else:
            df.loc[i,"Current_Score"]=df.loc[i-1,"Current_Score"]+df.loc[i,"total_runs"]

    columns_to_remove = ["total_runs"]
    df.drop(labels=columns_to_remove, axis=1, inplace=True)

    encoded_df = pd.get_dummies(data=df, columns=['batting_team', 'bowling_team'])
    encoded_df = encoded_df[['match_id', 'batsman', 'non_striker', 'bowler', 'Final_Score', 'Wickets',
    'over', 'ball', 'Current_Score',
    'batting_team_Chennai Super Kings', 'batting_team_Deccan Chargers',
    'batting_team_Delhi Daredevils', 'batting_team_Gujarat Lions',
    'batting_team_Kings XI Punjab',
    'batting_team_Kolkata Knight Riders', 'batting_team_Mumbai Indians',
    'batting_team_Pune Warriors', 'batting_team_Rajasthan Royals',
    'batting_team_Rising Pune Supergiant',
    'batting_team_Royal Challengers Bangalore',
    'batting_team_Sunrisers Hyderabad', 'bowling_team_Chennai Super Kings',
    'bowling_team_Deccan Chargers', 'bowling_team_Delhi Daredevils',
    'bowling_team_Gujarat Lions', 'bowling_team_Kings XI Punjab',
    'bowling_team_Kochi Tuskers Kerala',
    'bowling_team_Kolkata Knight Riders', 'bowling_team_Mumbai Indians',
    'bowling_team_Pune Warriors', 'bowling_team_Rajasthan Royals',
    'bowling_team_Rising Pune Supergiant',
    'bowling_team_Rising Pune Supergiants',
    'bowling_team_Royal Challengers Bangalore',
    'bowling_team_Sunrisers Hyderabad']]

    
    x = encoded_df.iloc[:,5:].values
    y = encoded_df.iloc[:,4].values

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x= sc.fit_transform(x)

    from sklearn.ensemble import RandomForestRegressor
    lin = RandomForestRegressor(n_estimators=100,max_features=None)
    lin.fit(x,y)

    dft['Wickets']=0
    dft["Final_Score"]=0
    dft["Current_Score"]=0
    dft.replace(np.NaN, int(0), inplace=True)
    for i in range(1,dft.shape[0]):
        if dft.loc[i, "match_id"] != dft.loc[i-1,"match_id"]:
            dft.loc[i,"Wickets"]=0
            if dft.loc[i, "player_dismissed"] != 0:
                dft.loc[i,"Wickets"]=1
        else:
            if dft.loc[i, "player_dismissed"] != 0:
                dft.loc[i,"Wickets"]=dft.loc[i-1,"Wickets"]+1
            else:
                dft.loc[i,"Wickets"]=dft.loc[i-1,"Wickets"]
    
    columns_to_remove = ["player_dismissed","dismissal_kind","fielder", "wide_runs","bye_runs","legbye_runs", "penalty_runs", "noball_runs","batsman_runs","extra_runs"]
    dft.drop(labels=columns_to_remove, axis=1, inplace=True)

    for i in range(1,dft.shape[0]):
        if dft.loc[i, "match_id"] != dft.loc[i-1,"match_id"]:
            dft.loc[i,"Final_Score"]= dft.loc[i,"total_runs"]
        else:
            dft.loc[i,"Final_Score"]=dft.loc[i-1,"Final_Score"]+dft.loc[i,"total_runs"]

    for i in range(dft.shape[0]-2,-1,-1):
        if dft.loc[i, "match_id"] != dft.loc[i+1,"match_id"]:
            dft.loc[i,"Final_Score"]= dft.loc[i,"Final_Score"]
        else:
            dft.loc[i,"Final_Score"]=dft.loc[i+1,"Final_Score"]

    for i in range(1,dft.shape[0]):
        if dft.loc[i, "match_id"] != dft.loc[i-1,"match_id"]:
            dft.loc[i,"Current_Score"]= dft.loc[i,"total_runs"]
        else:
            dft.loc[i,"Current_Score"]=dft.loc[i-1,"Current_Score"]+dft.loc[i,"total_runs"]

    columns_to_remove = ["total_runs"]
    dft.drop(labels=columns_to_remove, axis=1, inplace=True)

    encoded_dft = pd.get_dummies(data=dft, columns=['batting_team', 'bowling_team'])
    encoded_dft = encoded_dft[['match_id', 'batsman', 'non_striker', 'bowler', 'Final_Score', 'Wickets',
    'over', 'ball', 'Current_Score',
    'batting_team_Chennai Super Kings', 'batting_team_Deccan Chargers',
    'batting_team_Delhi Daredevils', 'batting_team_Gujarat Lions',
    'batting_team_Kings XI Punjab',
    'batting_team_Kolkata Knight Riders', 'batting_team_Mumbai Indians',
    'batting_team_Pune Warriors', 'batting_team_Rajasthan Royals',
    'batting_team_Rising Pune Supergiant',
    'batting_team_Royal Challengers Bangalore',
    'batting_team_Sunrisers Hyderabad', 'bowling_team_Chennai Super Kings',
    'bowling_team_Deccan Chargers', 'bowling_team_Delhi Daredevils',
    'bowling_team_Gujarat Lions', 'bowling_team_Kings XI Punjab',
    'bowling_team_Kochi Tuskers Kerala',
    'bowling_team_Kolkata Knight Riders', 'bowling_team_Mumbai Indians',
    'bowling_team_Pune Warriors', 'bowling_team_Rajasthan Royals',
    'bowling_team_Rising Pune Supergiant',
    'bowling_team_Rising Pune Supergiants',
    'bowling_team_Royal Challengers Bangalore',
    'bowling_team_Sunrisers Hyderabad']]

    x_test = encoded_dft.iloc[:,5:].values
    
    y_pred = lin.predict(x_test)
    store=[]
    for i in range(0,len(encoded_dft)-1):
        if (encoded_dft["match_id"][i] != encoded_dft["match_id"][i+1]):
            store.append(y_pred[i])
    store.append(y_pred[i])

    np.savetxt("firstarray.csv", store , delimiter=",")


inningFile = os.path.join(os.getcwd(),'IPL_train.csv')
predictionFile = os.path.join(os.getcwd(),'IPL_test.csv')
iplprediction(inningFile, predictionFile)




