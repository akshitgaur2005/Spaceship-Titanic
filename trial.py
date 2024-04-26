import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from torch import nn, optim


train_data = pd.read_csv("./data/train.csv")
hot_enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ord_enc = OrdinalEncoder()
scaler = StandardScaler()

def conv_int(x):
    if x != "NaN":
        return x

def trans(X):
    return (X > 0.5)

train_data[["GroupId", "IndiId"]] = train_data["PassengerId"].str.split("_", expand=True)
train_data[["Deck", "CabinNum", "Side"]] = train_data["Cabin"].str.split("/", expand=True)
train_data[["FirstName","LastName"]] = train_data["Name"].str.rsplit(" ", n=1, expand=True)
train_data["CryoSleep"] = train_data["CryoSleep"].apply(lambda x: 1 if x == True else 0)
train_data["Transported"] = train_data["Transported"].apply(lambda x: 1 if x == True else 0)
train_data["VIP"] = train_data["VIP"].apply(lambda x: 1 if x == True else 0)

train_data["GroupId"] = train_data["GroupId"].apply(int)
train_data["CabinNum"] = train_data["CabinNum"].fillna(train_data["CabinNum"].mode()[0]).apply(int)

for col in ["HomePlanet", "Destination", "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Deck", "Side", "LastName"]:
    train_data[col] = train_data[col].fillna(train_data[col].mode()[0])

train_data.drop(columns=["PassengerId", "IndiId", "Cabin", "Name", "FirstName"], inplace=True)

objects = list(train_data.select_dtypes(include=["object"]).columns)
objects.remove("LastName")
lastname = pd.DataFrame(train_data["LastName"])

temp_data1 = hot_enc.fit_transform(train_data[objects])
temp_data1_cols = hot_enc.get_feature_names_out()
temp_data2 = ord_enc.fit_transform(lastname)
temp_data2_cols = ord_enc.get_feature_names_out()

n_data = train_data.drop(columns=[*objects, "LastName"])

data = pd.concat((n_data, pd.DataFrame(temp_data1, columns=temp_data1_cols), pd.DataFrame(temp_data2, columns=temp_data2_cols)), axis=1)
cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "GroupId", "CabinNum", "LastName"]
data[cols] = scaler.fit_transform(data[cols])
#print(train_data.info())
#print(train_data.nunique())
#print(train_data.head(1))

print(data)
X = data.drop("Transported", axis=1).to_numpy()
y = data["Transported"].to_numpy().reshape(-1, 1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

X = torch.from_numpy(X_train).to("cuda", dtype=torch.float)
y = torch.from_numpy(y_train).to("cuda", dtype=torch.float)
X_val = torch.from_numpy(X_val).to("cuda", dtype=torch.float)
y_val = torch.from_numpy(y_val).to("cuda", dtype=torch.float)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                nn.Linear(X.shape[1], 27),
                nn.ReLU(),
                nn.LazyLinear(27), nn.ReLU(),
                nn.LazyLinear(27), nn.ReLU(),
                nn.LazyLinear(1),
                nn.Sigmoid()
                )
    def forward(self, X):
        logits = self.layers(X)
        return logits

model = Model().to("cuda")
print(model)

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

tracker = 0
tol=5
val_loss = [1000]


for i in range(150000):
    pred = model(X)
    loss = loss_fn(pred, y)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    with torch.no_grad():
        val_preds = model(X_val)
        val_loss.append(loss_fn(val_preds, y_val))
        
        if val_loss[-1] > val_loss[-2]:
            tracker += 1
        else:
            tracker = 0
    fin_preds = trans(val_preds.cpu()) 
    if (i % 100 == 0):
        print(f"i: {i}, loss: {loss.item()}, val_loss: {val_loss[-1]}, acc: {accuracy_score(fin_preds, y_val.cpu())}")
    if tracker > tol:
        break

train_data = pd.read_csv("./data/test.csv")
ids = train_data["PassengerId"].values
train_data[["GroupId", "IndiId"]] = train_data["PassengerId"].str.split("_", expand=True)
train_data[["Deck", "CabinNum", "Side"]] = train_data["Cabin"].str.split("/", expand=True)
train_data[["FirstName","LastName"]] = train_data["Name"].str.rsplit(" ", n=1, expand=True)
train_data["CryoSleep"] = train_data["CryoSleep"].apply(lambda x: 1 if x == True else 0)
train_data["VIP"] = train_data["VIP"].apply(lambda x: 1 if x == True else 0)

train_data["GroupId"] = train_data["GroupId"].apply(int)
train_data["CabinNum"] = train_data["CabinNum"].fillna(train_data["CabinNum"].mode()[0]).apply(int)

for col in ["HomePlanet", "Destination", "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Deck", "Side", "LastName"]:
    train_data[col] = train_data[col].fillna(train_data[col].mode()[0])

train_data.drop(columns=["PassengerId", "IndiId", "Cabin", "Name", "FirstName"], inplace=True)

objects = list(train_data.select_dtypes(include=["object"]).columns)
objects.remove("LastName")
lastname = pd.DataFrame(train_data["LastName"])

temp_data1 = hot_enc.fit_transform(train_data[objects])
temp_data1_cols = hot_enc.get_feature_names_out()
temp_data2 = ord_enc.fit_transform(lastname)
temp_data2_cols = ord_enc.get_feature_names_out()

n_data = train_data.drop(columns=[*objects, "LastName"])

data = pd.concat((n_data, pd.DataFrame(temp_data1, columns=temp_data1_cols), pd.DataFrame(temp_data2, columns=temp_data2_cols)), axis=1)
cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "GroupId", "CabinNum", "LastName"]
data[cols] = scaler.fit_transform(data[cols])

X = torch.from_numpy(data.to_numpy()).to("cuda", dtype=torch.float)
preds = model(X)


new_preds= trans(preds.cpu())

result = pd.DataFrame({
    "PassengerId": ids,
    "Transported": new_preds.reshape(-1)
    })

print(result)

result.to_csv("result.csv", index=False)
