import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from io import StringIO

@st.cache
def load_model(uploaded_file):
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    df = pd.read_csv(stringio, names=['top-left', 'top-middle', 'top-right', 'middle-left', 'middle-middle', 'middle-right', 'bottom-left', 'bottom-middle', 'bottom-right', 'class'])

    ohe = OneHotEncoder(sparse_output=False)  
    board_columns = ['top-left', 'top-middle', 'top-right', 'middle-left', 'middle-middle', 'middle-right', 'bottom-left', 'bottom-middle', 'bottom-right']
    encoded_board = ohe.fit_transform(df[board_columns])

    df['class'] = df['class'].map({'positive': 1, 'negative': 0})

    encoded_df = pd.DataFrame(encoded_board, columns=ohe.get_feature_names_out(board_columns))
    df = pd.concat([encoded_df, df['class']], axis=1)

    X = df.iloc[:, :-1]
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, ohe

def train_predict(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def tune_hyperparameters(X_train, y_train):
    # Grid Search for Hyperparameter Tuning (for RandomForest)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_

st.title("Tic-Tac-Toe Game Prediction")
st.markdown("""
    **Dataset Description**:
    This database encodes the complete set of possible board configurations at the end of tic-tac-toe games, where "x" is assumed to have played first. 
    The target concept is "win for x" (i.e., true when "x" has one of 8 possible ways to create a "three-in-a-row").
    The dataset has 9 features representing the Tic-Tac-Toe board and a target column indicating whether 'x' wins or not.
""")
st.write("ฐานข้อมูลนี้เข้ารหัสชุดค่าที่เป็นไปได้ทั้งหมดของการจัดวางกระดานในตอนจบของเกม Tic-Tac-Toe "
"โดยสมมติให้ x เป็นผู้เล่นที่เล่นก่อนแนวคิดเป้าหมายคือ ชัยชนะของ x (เป็นจริงเมื่อ x ชนะด้วยวิธีการเรียงสามช่องติดกันในแนวตั้ง แนวนอน หรือแนวทแยง 8 รูปแบบที่เป็นไปได้)"
"ชุดข้อมูลนี้มี 9 คุณลักษณะที่แทนตำแหน่งบนกระดาน Tic-Tac-Toe และคอลัมน์เป้าหมายที่ระบุว่า x ชนะหรือไม่")

st.write("b for blank, x for x, o for o")
st.title("สมมุติว่าคุณเป็น x ในเกม Tic-Tac-Toe และต้องการทราบว่าคุณจะชนะหรือไม่ และคุณเริ่มก่อน")
uploaded_file = st.file_uploader("Choose a .data file", type=["data"])

if uploaded_file is not None:
    X_train, X_test, y_train, y_test, ohe = load_model(uploaded_file)

    model_options = ["Decision Tree", "K-Nearest Neighbors", "Random Forest", "Gradient Boosting"]
    selected_model = st.sidebar.selectbox("Choose a model", model_options)

    if selected_model == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif selected_model == "K-Nearest Neighbors":
        model = KNeighborsClassifier(n_neighbors=3)
    elif selected_model == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif selected_model == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)

    if selected_model == "Random Forest":
        model = tune_hyperparameters(X_train, y_train)

    accuracy = train_predict(X_train, X_test, y_train, y_test, model)
    st.write(f"Model Accuracy: {accuracy:.4f}")

    st.sidebar.header("Input Game Features")
    top_left = st.sidebar.selectbox("Top-left", ["x", "o", "b"])
    top_middle = st.sidebar.selectbox("Top-middle", ["x", "o", "b"])
    top_right = st.sidebar.selectbox("Top-right", ["x", "o", "b"])
    middle_left = st.sidebar.selectbox("Middle-left", ["x", "o", "b"])
    middle_middle = st.sidebar.selectbox("Middle-middle", ["x", "o", "b"])
    middle_right = st.sidebar.selectbox("Middle-right", ["x", "o", "b"])
    bottom_left = st.sidebar.selectbox("Bottom-left", ["x", "o", "b"])
    bottom_middle = st.sidebar.selectbox("Bottom-middle", ["x", "o", "b"])
    bottom_right = st.sidebar.selectbox("Bottom-right", ["x", "o", "b"])
    features = [
        *ohe.transform([[top_left, top_middle, top_right, middle_left, middle_middle, middle_right, bottom_left, bottom_middle, bottom_right]]).flatten()
    ]

    if st.sidebar.button("Predict"):
        prediction = model.predict([features])
        result = "Win for x" if prediction == 1 else "No win for x"
        st.write(f"The predicted result is: {result}")
