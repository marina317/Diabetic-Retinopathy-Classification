from sklearn.ensemble import RandomForestClassifier

def build_model(model_name = "random_forest"):
  if model_name == "random_forest":
    model = RandomForestClassifier(n_estimators = 100, max_depth = 20, class_weight = 'balanced', random_state = 42)
    return model
  elif model_name == "svm":
    model = SVC(kernel = "rbf", class_weight = 'balanced', random_state = 42)
    return model
  elif model_name == "decision_tree":
    model = DecisionTreeClassifier(max_depth20, class_weight = 'balanced', random_state = 42)
    return model
  else:
    raiseValueError("unkown model name")

def train_pipeline(model_name, df):
  x = np.vstack(df["features"].values)
  y = df["label"]
  encoder = LabelEncoder()
  y = encoder.fit_transform(y)
  x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y)
  scaler = StandardScaler()
  x_train = scaler.fit_transform(x_train)
  x_test = scaler.transform(x_test) 
  model = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    max_depth=20, 
    class_weight='balanced'
)
  model.fit(x_train, y_train)
  return model


  
  

def evaluate_model():
  pass

def save_model():
  pass
    
