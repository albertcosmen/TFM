import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objs as go
import shap
import matplotlib.pyplot as plt
from plotly.offline import iplot
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score, roc_auc_score
from geopy.geocoders import Nominatim


@st.cache_data
def load_data(filename):
    keep_columns =[
        "iyear", 
        "imonth", 
        "iday", 
        "country_txt",
        "city",
        "region_txt",
        "latitude",
        "longitude", 
        "provstate",
        "targtype1_txt",
        "attacktype1_txt",
        "gname",
        "nkill",
        "nwound",
        "country",
        "region",
        "suicide",
        "attacktype1",
        "targtype1",
        "weaptype1",
        "weaptype1_txt",
        "success",
        "eventid",
        "natlty1",
        "natlty1_txt",
        "extended",
        "specificity",
        "vicinity",
        "crit1",
        ]

    df = pd.read_csv(filename,encoding="latin-1")  
    df = df[keep_columns]
    return df

@st.cache_resource
def train_model_xgb(dfnew):
    dfnew = dfnew.dropna()
    X = dfnew.drop(["success"], axis=1, inplace=False)
    Y = dfnew["success"]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=100)
    # Calcula peso para la clase minoritaria
    neg, pos = np.bincount(Y)
    scale = neg / pos
    classifier = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale)
    classifier.fit(X_train, y_train)
        
    y_pred = classifier.predict(X_test)
    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
    prec = round(precision_score(y_test, y_pred) * 100, 2)
    rec = round(recall_score(y_test, y_pred) * 100, 2)

    return classifier, acc, prec, rec


def world_line_attacks_over_time(df):

    data = df["iyear"].value_counts().to_frame().reset_index()
    data.columns = ["Year", "Number of Attacks"]
    data.sort_values(by = "Year", inplace = True)

    country_trace = go.Scatter(
        x = data["Year"],
        y = data["Number of Attacks"],
        mode = "lines+markers",
        marker = dict(color="red")
    )

    Layout = dict(
        margin=dict(l=0,r=0,b=0,t=0),
        width = 700,
        height = 250,
        xaxis=dict(title="Year"), 
        yaxis=dict(title="Number of attacks"),
    )
    
    fig = go.Figure(
        data= country_trace,
        layout=go.Layout( Layout)
    )

    return fig

def region_line_attacks_over_time(df,country):

    country_filter = df["country_txt"] == country
    filtered = df[country_filter]
    region = filtered.iloc[0]['region_txt']
    region_filter = df["region_txt"] == region
    filtered = df[region_filter]
    data = filtered["iyear"].value_counts().to_frame().reset_index()
    data.columns = ["Year", "Number of Attacks"]
    data.sort_values(by = "Year", inplace = True)

    country_trace = go.Scatter(
        x = data["Year"],
        y = data["Number of Attacks"],
        mode = "lines+markers",
        marker = dict(color="red")
    )

    Layout = dict(
        margin=dict(l=0,r=0,b=0,t=0),
        width = 700,
        height = 250,
        xaxis=dict(title="Year"), 
        yaxis=dict(title="Number of attacks"),
    )
    
    fig = go.Figure(
        data= country_trace,
        layout=go.Layout( Layout)
    )

    return fig, region



def line_attacks_over_time(df,country):

    country_filter = df["country_txt"] == country
    filtered = df[country_filter]
    data = filtered["iyear"].value_counts().to_frame().reset_index()
    data.columns = ["Year", "Number of Attacks"]
    data.sort_values(by = "Year", inplace = True)

    country_trace = go.Scatter(
        x = data["Year"],
        y = data["Number of Attacks"],
        mode = "lines+markers",
        marker = dict(color="red")
    )

    Layout = dict(
        margin=dict(l=0,r=0,b=0,t=0),
        width = 700,
        height = 250, 
        xaxis=dict(title="Year"), 
        yaxis=dict(title="Number of attacks"),
    )
    
    fig = go.Figure(
        data= country_trace,
        layout=go.Layout( Layout)
    )

    return fig

def pie_most_dangerous_cities (df,country):
    country_filter = df["country_txt"] == country
    filtered = df[country_filter]

    bad_cities = filtered["city"].value_counts().to_frame().reset_index()
    bad_cities.columns = ["City", "Number of Attacks"]
    total_attacks = bad_cities["Number of Attacks"].sum()
    try:
        thresh = bad_cities["Number of Attacks"].iloc[5]
    except:
        thresh = 0

    big_bad_cities = bad_cities[bad_cities["Number of Attacks"]> thresh]
    other = pd.Series({"City":"Other",
    "Number of Attacks":total_attacks - big_bad_cities["Number of Attacks"].sum()})

    if other["Number of Attacks"] !=0:
        big_bad_cities = pd.concat([big_bad_cities, other], ignore_index=True)

    Data = dict(
        values = big_bad_cities["Number of Attacks"],
        labels = big_bad_cities["City"],
        type = "pie",
        hole = 0.3,
        showlegend=False
    )
    Layout = dict(
        margin=dict(l=20,r=0,b=0,t=0),
        width = 400,
        height = 400,
    )
    
    fig = go.Figure(
        data= Data,
        layout=go.Layout( Layout)
    )

    return fig

def pie_most_attacked_targets (df,country):
    country_filter = df["country_txt"] == country
    filtered = df[country_filter]

    attacked_targets = filtered["targtype1_txt"].value_counts().to_frame().reset_index()
    attacked_targets.columns = ["Target", "Number of Attacks"]
    total_attacks = attacked_targets["Number of Attacks"].sum()

    try:
        thresh = attacked_targets["Number of Attacks"].iloc[6]
    except:
        thresh = 0

    main_target = attacked_targets[attacked_targets["Number of Attacks"]> thresh]
    other = pd.Series({"Target":"Other",
    "Number of Attacks":total_attacks - attacked_targets["Number of Attacks"].sum()})

    if other["Number of Attacks"] !=0:
        main_target = pd.concat([main_target, other], ignore_index=True)

    Data = dict(
        values = main_target["Number of Attacks"],
        labels = main_target["Target"],
        type = "pie",
        hole = 0.3,
        showlegend=False
    )
    Layout = dict(
        margin=dict(l=20,r=0,b=0,t=0),
        width = 400,
        height = 400,
    )
    
    fig = go.Figure(
        data= Data,
        layout=go.Layout( Layout)
    )


    return fig

def pie_most_freq_type_attack (df,country):
    country_filter = df["country_txt"] == country
    filtered = df[country_filter]

    attack_type = filtered["attacktype1_txt"].value_counts().to_frame().reset_index()
    attack_type.columns = ["Attack Type", "Number of Attacks"]
    total_attacks = attack_type["Number of Attacks"].sum()

    try:
        thresh = attack_type["Number of Attacks"].iloc[5]
    except:
        thresh = 0

    main_type_attack = attack_type[attack_type["Number of Attacks"]> thresh]
    other = pd.Series({"Attack Type":"Other",
    "Number of Attacks":total_attacks - attack_type["Number of Attacks"].sum()})

    if other["Number of Attacks"] !=0:
        main_type_attack = pd.concat([main_type_attack, other], ignore_index=True)

    Data = dict(
        values = main_type_attack["Number of Attacks"],
        labels = main_type_attack["Attack Type"],
        type = "pie",
        hole = 0.3,
        showlegend=False
    )
    Layout = dict(
        margin=dict(l=20,r=0,b=0,t=0),
        width = 400,
        height = 400,
    )
    
    fig = go.Figure(
        data= Data,
        layout=go.Layout( Layout)
    )

    return fig

def pie_most_active_groups (df,country):
    country_filter = df["country_txt"] == country
    filtered = df[country_filter]

    terror_groups = filtered["gname"].value_counts().to_frame().reset_index()
    terror_groups.columns = ["Terrorist Group", "Number of Attacks"]
    total_attacks = terror_groups["Number of Attacks"].sum()

    try:
        thresh = terror_groups["Number of Attacks"].iloc[5]
    except:
        thresh = 0

    main_terror_groups = terror_groups[terror_groups["Number of Attacks"]> thresh]
    other = pd.Series({"Terrorist Group":"Other",
    "Number of Attacks":total_attacks - terror_groups["Number of Attacks"].sum()})

    if other["Number of Attacks"] !=0:
        main_terror_groups = pd.concat([main_terror_groups, other], ignore_index=True)

    Data = dict(
        values = main_terror_groups["Number of Attacks"],
        labels = main_terror_groups["Terrorist Group"],
        type = "pie",
        hole = 0.3,
        showlegend=False
    )
    Layout = dict(
        margin=dict(l=20,r=0,b=0,t=0),
        width = 400,
        height = 400,
    )
    
    fig = go.Figure(
        data= Data,
        layout=go.Layout( Layout)
    )

    return fig



df = load_data("Terrorism_clean_dataset.csv")


st.title("Global Terrorism Exploration APP")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Historical Analysis",
    "🤖 ML Prediction",
    "🔮 Future Simulation",
    "📦 Export & Reports"
])
with tab1:
    #SideBar Region Selector
    regions = df["region_txt"].unique().tolist()
    regions.insert(0,"All Regions")
    region = st.sidebar.selectbox("Choose a Region",options =  regions)
    if region == "All Regions":
        in_region = df["region_txt"] == df["region_txt"]
    else:
        in_region = df["region_txt"] == region
    
    #SIDEBAR COUNTRY SELECTOR
    countries = df.loc[in_region, "country_txt"].unique()
    country= st.sidebar.selectbox("Choose a Country", options = countries)
    is_country = df["country_txt"] == country
    
    st.markdown("")
    st.header(f"Terrorism Brief: {country}")
    st.markdown("")
    st.markdown("")
    
    col_info, col_viz = st.columns ([1,1.3])
    
    ##Brief part column information
    with col_info:
        #Most Attacked City
        worst_city = df.loc[is_country,"city"].value_counts().head(1).index[0]
        city_in_state = df[df["city"] == worst_city].iloc[0]["provstate"]
        st.subheader("Most-Attacked City:")
        if type(city_in_state) != float:
            st.markdown(f"*{worst_city}, {city_in_state}*")
        else:
            st.markdown(f"*{worst_city}*")
    
        #Most Attacked Year
        worst_year = df.loc[is_country,"iyear"].value_counts().head(1).index[0]
        num_attacks_worst_year = df.loc[is_country,"iyear"].value_counts().iloc[0]
        st.subheader("Year of Most Attacks: ")
        st.markdown(f"*{worst_year} with {num_attacks_worst_year} attacks*")
    
        #Best Year
        best_year = df.loc[is_country,"iyear"].value_counts().tail(1).index[0]
        num_attacks_best_year = df.loc[is_country,"iyear"].value_counts().tail(1).values[0]
        st.subheader("Year of Fewest Attacks:")
        st.markdown(f"*{best_year} with {num_attacks_best_year} attacks*")
    
        #Most Mortal Year
        df_filtered = df[is_country]
        mortal_year_filtered = df_filtered.groupby(["iyear"]).agg(Total_attacks=("nkill","count"), Total_Killed=("nkill","sum"), Total_Wound=("nwound","sum")).sort_values(by=["Total_Killed"],ascending=False).reset_index()
        year_mortal = mortal_year_filtered.iloc[0]["iyear"]
        num_kill_worst_year =  mortal_year_filtered.iloc[0]['Total_Killed']
        st.subheader("Year of Most people killed: ")
        st.markdown(f"*{year_mortal} with {num_kill_worst_year} people killed*")
    
        #Most Dangerous Terrorist Group
        terrorist_group_filtered = df_filtered.groupby(["gname"]).agg(Total_attacks=("nkill","count"), Total_Killed=("nkill","sum"), Total_Wound=("nwound","sum")).sort_values(by=["Total_Killed"],ascending=False).reset_index()
        worst_group = terrorist_group_filtered.iloc[0]["gname"]
        if worst_group == "Unknown":
            worst_group = terrorist_group_filtered.iloc[1]["gname"]
            num_kill_t_worst_year =  terrorist_group_filtered.iloc[1]['Total_Killed']
        else:
            num_kill_t_worst_year =  terrorist_group_filtered.iloc[0]['Total_Killed']
    
        st.subheader("The Most dangerous Terrorist Group: ")
        st.markdown(f"**{worst_group}** *with* **{num_kill_t_worst_year} people killed** *in total during all the period (1970-2017)*")
    
    
    isyear=df["iyear"].unique().tolist()
    isyear.insert(0,"All history")
    
    ## Brief part, plot selection option
    with col_viz:
        plot_type = st.selectbox("Choose a visualization:", options = ["Pie: Most Dangerous Cities","Pie: Most Attacked Targets","Pie: Most Frequent Type of Attack", "Pie: Main Terrorist Groups"])
    
        if plot_type == "Pie: Most Dangerous Cities":
            st.plotly_chart(pie_most_dangerous_cities(df,country), width=300 , height=400, margin=dict(l=0, r=0, b=0, t=0),autosize=False,)
            worst_cities = df.loc[is_country,"city"].value_counts().head(5).rename_axis('City').reset_index(name='Total Attacks')
            st.subheader("Most-Dangerous Cities:")
            st.dataframe(worst_cities)
    
        elif plot_type == "Pie: Most Attacked Targets":
            st.plotly_chart(pie_most_attacked_targets(df,country),width=300 , height=400, margin=dict(l=0, r=0, b=0, t=0),autosize=False,)
            worst_targets = df.loc[is_country,"targtype1_txt"].value_counts().head(5).rename_axis('Target Type').reset_index(name='Total Attacks')
            st.subheader("Most Attacked Targets: ")
            st.dataframe(worst_targets)
    
        elif plot_type == "Pie: Most Frequent Type of Attack":
            st.plotly_chart(pie_most_freq_type_attack(df,country),width=300 , height=400, margin=dict(l=0, r=0, b=0, t=0),autosize=False,)
            freq_attack = df.loc[is_country,"attacktype1_txt"].value_counts().head(5).rename_axis('Attack Type').reset_index(name='Total Attacks')
            st.subheader("Most frequent type of attack: ")
            st.dataframe(freq_attack)
    
        elif plot_type == "Pie: Main Terrorist Groups":
            st.plotly_chart(pie_most_active_groups(df,country),width=300 , height=400, margin=dict(l=0, r=0, b=0, t=0),autosize=False,)
            worst_groups = df.loc[is_country,"gname"].value_counts().head(5).rename_axis('Group Name').reset_index(name='Total Attacks')
            st.subheader("Most Active Terrorist Groups: ")
            st.dataframe(worst_groups)
    
    st.markdown("")
    st.subheader(f"{country}: Nationwide Attacks over Time")
    st.markdown(f"The following chart represents the **total attacks per year from 1970 till 2017** in **{country}**")
    st.markdown("")
    st.plotly_chart(line_attacks_over_time(df,country))
    
    
    figure, region1 = region_line_attacks_over_time(df,country)
    st.markdown("")
    st.subheader(f"{region1}: Attacks over Time")
    st.markdown(f"The following chart represents the **total attacks per year from 1970 till 2017** accross **{region1} region**, the goal is to have as a reference to compare with, in order to see if {country} have a terrorist activity inusual regarding the region, if is a local problem, or a regional problem. ")
    st.markdown("")
    st.plotly_chart(figure)
    
    st.markdown("")
    st.subheader("Worldwide Attacks over Time")
    st.markdown(f"The following chart represents the **total attacks per year from 1970 till 2017 Worldwide**")
    st.markdown("")
    st.plotly_chart(world_line_attacks_over_time(df))
    
    
    st.markdown("")
    st.markdown("")
    st.header(f"Exploring data: {country}")
    st.markdown("")
    st.markdown("")
    st.warning("Please, choose a range of time using the slidebar on the left Menu to explore the data below.")
    st.markdown("")
    
    y_min =min(df["iyear"])
    y_max =max(df["iyear"])
    y1,y2 = st.sidebar.slider("Choose a range:", y_min, y_max, (y_min, y_max))
    
    # --- Prediction Threshold Slider ---
    st.sidebar.markdown("### 🔧 Prediction Threshold")
    threshold = st.sidebar.slider(
        "Minimum confidence for predicting 'Success':",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05
    )
    
    in_year_range = df["iyear"].isin(range(y1,y2+1))
    
    st.subheader(f"{country}: Map representation with all_attacks between ({y1} - {y2})")
    st.markdown(f"With regards to the period choosen, in the map is represented the attacks distributed accross the territory:")
    
    
    map_data = df.dropna(axis=0, subset=["latitude", "longitude"])
    in_lat = (df["latitude"] >=-90) & (df["latitude"] <=90)
    in_lon = (df["longitude"] >=-180) & (df["longitude"]<=180)
    map_data=map_data[in_lat & in_lon & is_country & in_year_range]
    
    st.map(map_data)
    df_filtered = df[in_year_range & is_country]
    terror_groups_filtered = df_filtered.groupby(["gname"]).agg(Total_attacks=("nkill","count"), Total_Killed=("nkill","sum"), Total_Wound=("nwound","sum")).reset_index()
    terror_groups_filtered = terror_groups_filtered.rename(columns={"gname":"Group Name"})
    
    ###################################################
    st.markdown("")
    st.markdown("")
    st.subheader(f"Terrorist Groups in {country} between {y1} and {y2}: ")
    st.markdown(f"With regards to the period choosen, in the following table shows the terrorist groups active in {country} between {y1} and {y2}, showing the total attacks committed, number of people killed and wound by each group.")
    st.dataframe(terror_groups_filtered)
    st.markdown("")
    st.markdown("")
    
    ###################################################
    attacks_type_filtered = df_filtered.groupby(["attacktype1_txt"]).agg(Total_attacks=("nkill","count"), Total_Killed=("nkill","sum"), Total_Wound=("nwound","sum")).reset_index()
    st.subheader(f"Terrorist Attack types in {country} between {y1} and {y2}: ")
    st.markdown(f"With regards to the period choosen, in the following table shows the types of terrorist attacks in {country} between {y1} and {y2}, with the total attacks committed, number of people killed and wound by each type")
    
    fig = go.Figure(data=[go.Histogram(x=df_filtered["attacktype1_txt"],marker = dict(color = "red"))], layout= dict (title=f"{country}: Terrorist attacks types  ({y1} - {y2})", xaxis=dict(title="Terrorist attack Types"), yaxis=dict(title="Total number of attacks")))
    st.plotly_chart(fig,width=600 , height=400, margin=dict(l=0, r=0, b=0, t=0))
    st.markdown("")
    st.markdown("")
    
    attacks_type_filtered= attacks_type_filtered.rename(columns={"attacktype1_txt":"Type of Attack"})
    st.markdown("")
    st.markdown("")
    st.markdown("In the table below, the more detailed information:")
    st.dataframe(attacks_type_filtered)
    st.markdown("")
    st.markdown("")
    
    st.subheader(f"Terrorist Attack targets in {country} between {y1} and {y2}: ")
    st.markdown(f"With regards to the period choosen, in the following table shows the terrorist attack targets in {country} between {y1} and {y2}, with the total attacks committed, number of people killed and wound by each type")
    
    
    fig = go.Figure(data=[go.Histogram(x=df_filtered["targtype1_txt"],marker = dict(color = "red"))], layout= dict (title=f"{country}: Terrorist attacks targets ({y1} - {y2})", xaxis=dict(title="Terrorist attack targets"), yaxis=dict(title="Total number of attacks")))
    st.plotly_chart(fig,width=600 , height=400, margin=dict(l=0, r=0, b=0, t=0))
    
    
    attacks_targets_filtered = df_filtered.groupby(["targtype1_txt"]).agg(Total_attacks=("nkill","count"), Total_Killed=("nkill","sum"), Total_Wound=("nwound","sum")).reset_index()
    attacks_targets_filtered= attacks_targets_filtered.rename(columns={"targtype1_txt":"Type of Target"})
    st.markdown("")
    st.markdown("")
    st.markdown("In the table below, the more detailed information:")
    st.dataframe(attacks_targets_filtered)
    st.markdown("")
    st.markdown("")
    
with tab2:
    ###************************MACHINE LEARNING**********************************###
    
    st.header(f"{country}: Machine Learning, terrorism success attack prediction")
    st.markdown("")
    st.markdown("")
    
    #Choosing Variables for feed our  model
    
    dfnew = df[["imonth","iday", "success","attacktype1","targtype1","natlty1",
                "weaptype1","nkill","nwound","region","latitude","longitude",
                "specificity","vicinity","extended","suicide"]]
    
    dfnew = dfnew.dropna()
    
    # 🔁 Escalar muertes y heridos (log-transformación + 1 para evitar log(0))
    dfnew["nkill"] = dfnew["nkill"].apply(lambda x: np.log1p(x))
    dfnew["nwound"] = dfnew["nwound"].apply(lambda x: np.log1p(x))
    
    X = dfnew.drop(["success"], axis=1, inplace = False)
    Y = dfnew["success"]
    
    ##Spliting data set in training and test
    X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
    
    dfnew = dfnew.drop(columns=["lab_kill", "lab_wound"], errors="ignore")
    classifier, acc, prec, rec = train_model_xgb(dfnew)
    classifier.fit(X_train, y_train)
    
    # --- Importancia global de variables ---
    importances = classifier.feature_importances_
    features = X.columns
    
    # Crear DataFrame y ordenar
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(5)
    
    st.subheader("🔝 Top 5 Most Important Features (Global)")
    st.table(importance_df)
    
    # También mostrar como gráfico de barras
    fig_imp, ax_imp = plt.subplots()
    ax_imp.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
    ax_imp.invert_yaxis()
    ax_imp.set_xlabel("Importance")
    ax_imp.set_title("Top 5 Features by Model Importance")
    
    st.pyplot(fig_imp)
    
    # --- Validación cruzada (scoring = accuracy) ---
    with st.spinner("Evaluating model with 5-fold cross-validation..."):
        cv_scores = cross_val_score(classifier, X, Y, cv=5, scoring="accuracy")
        st.markdown("### 🧪 Cross-validation accuracy scores:")
        st.write(cv_scores)
        st.markdown(f"**Mean Accuracy (5-fold):** {cv_scores.mean():.2%}")
    
    ##Predict Y with the x test
    y_pred = classifier.predict(X_test)
    
    # --- Matriz de confusión ---
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(cm).plot(ax=ax_cm)
    st.subheader("Confusion Matrix (Test Set)")
    st.pyplot(fig_cm)
    
    # --- Curva ROC ---
    fpr, tpr, _ = roc_curve(y_test, classifier.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax_roc.set_xlim([0.0, 1.0])
    ax_roc.set_ylim([0.0, 1.05])
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('ROC Curve')
    ax_roc.legend(loc="lower right")
    
    st.subheader(" ROC Curve (Test Set)")
    st.pyplot(fig_roc)
    
    ## Calculation of accuracy score
    ac = str(acc)
    
    st.markdown("### Model Performance Metrics:")
    st.table(pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall"],
        "Value (%)": [acc, prec, rec]
    }))
    
    # Cálculo de F1 y AUC
    f1 = round(f1_score(y_test, y_pred) * 100, 2)
    roc_auc = round(roc_auc_score(y_test, classifier.predict_proba(X_test)[:, 1]) * 100, 2)
    
    # Mostrar en tabla adicional
    st.table(pd.DataFrame({
        "Metric": ["F1-Score", "ROC-AUC"],
        "Value (%)": [f1, roc_auc]
    }))
    
    # Calcular precision y recall para muchos thresholds
    probas = classifier.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, probas)
    
    # Gráfico Precision vs Recall según threshold
    fig_pr, ax_pr = plt.subplots()
    ax_pr.plot(thresholds, precisions[:-1], label="Precision", color='green')
    ax_pr.plot(thresholds, recalls[:-1], label="Recall", color='blue')
    ax_pr.set_xlabel("Threshold")
    ax_pr.set_ylabel("Score")
    ax_pr.set_title("Precision vs Recall Curve")
    ax_pr.legend()
    ax_pr.grid(True)
    
    st.subheader("🎛 Precision / Recall vs Threshold")
    st.pyplot(fig_pr)
    
    
    
    ##Function to get the imputs for predict
    def user_report(df, dfclean, region, country):
    
        geolocator = Nominatim(user_agent="my_user_agent")
    
        cities =pd.unique(df[df['country_txt'] == country ]['city'].sort_values()).tolist()
        city = st.selectbox("Choose a City",options =  cities)
    
        loc = geolocator.geocode(city+','+ country)
    
        imonthst = dfclean["imonth"].sort_values().unique().tolist()
        imonths = st.selectbox("Choose month:",options = imonthst)
    
        idayst = dfclean["iday"].sort_values().unique().tolist()
        idays = st.selectbox("Choose day:",options =  idayst)
        
        Attacktext = pd.unique(df['attacktype1_txt'].sort_values()).tolist()
        Attacktext = st.selectbox("Choose the Type of Attack",options =  Attacktext)
        attacktype1=df.loc[df['attacktype1_txt'] == Attacktext, 'attacktype1'].iloc[0]
        
        Targettext = pd.unique(df['targtype1_txt'].sort_values()).tolist()
        Targettext= st.selectbox("Choose a Target",options =  Targettext)
        targettype1=df.loc[df['targtype1_txt'] == Targettext, 'targtype1'].iloc[0]
    
        Weapntext = pd.unique(df['weaptype1_txt'].sort_values()).tolist()
        Weapntext = st.selectbox("Choose the main Weapon used in the attack",options =  Weapntext)
        weaptype = df.loc[df['weaptype1_txt'] == Weapntext, 'weaptype1'].iloc[0]
        
        nattext = pd.unique(df['natlty1_txt'].sort_values()).tolist()
        nattext = st.selectbox("Choose the target nationality",options =  nattext)
        natlty1= df.loc[df['natlty1_txt'] == nattext, 'natlty1'].iloc[0]
    
        
        killed = st.number_input("Number of people killed", min_value=0, max_value=500, step=1)
    
        twound = st.number_input("Number of people wounded", min_value=0, max_value=1000, step=1)
        
        region = df.loc[df['region_txt'] == region, 'region'].iloc[0]
        latitude = loc.latitude
        longitude = loc.longitude
    
        Specificity = dfclean["specificity"].sort_values().unique().tolist()
        Specificity = st.selectbox("Choose the Specificity", options =  Specificity)
        specificity = Specificity
        st.markdown("**Specificity:** **1** = event occurred in city/village/town and lat/long is for that location, **2** = event occurred in city/village/town and no lat/long could be found, so coordinates are for centroid of smallest subnational administrative region identified, **3** = event did not occur in city/village/town, so coordinates are for centroid of smallest subnational administrative region identified, **4** = no 2nd order or smaller region could be identified, so coordinates are for center of 1st order administrative region,**5** = no 1st order administative region could be identified for the location of the attack, so latitude and longitude are unknown")
    
        Vicinity = dfclean["vicinity"].sort_values().unique().tolist()
        Vicinity = st.selectbox("Choose if Vicinity (0 - No, 1 - Yes)", (0,1))
        vicinity = Vicinity
    
        Extended = dfclean["extended"].sort_values().unique().tolist()
        Extended = st.selectbox("Choose if Extended attack (0 - No, 1 - Yes)", options =  Extended)
        extended = Extended
    
        Suicide = dfclean["suicide"].sort_values().unique().tolist()
        Suicide = st.selectbox("Choose if Suicide Attack (0 - No, 1 - Yes)", options =  Suicide)
        suicide = Suicide
    
        user_report = {
    
            "imonth":imonths,
            "iday":idays,
            "attacktype1":attacktype1,
            "targtype1":targettype1,
            "natlty1":natlty1,
            "weaptype1":weaptype,
            "region": region,
            "latitude": latitude,
            "longitude": longitude,
            "specificity": specificity,
            "vicinity":vicinity,
            "extended": extended,
            "suicide":suicide,
            "nkill": killed,
            "nwound": twound,
        }
    
        report_data = pd.DataFrame(user_report, index=[0])
    
        # Orden y filtrado exacto de columnas usadas en el modelo
        model_features = ['imonth', 'iday', 'attacktype1', 'targtype1', 'natlty1',
                          'weaptype1', 'region', 'latitude', 'longitude', 'specificity',
                          'vicinity', 'extended', 'suicide', 'nkill', 'nwound']
        report_data = report_data[model_features]
        
        return report_data
    
    user_data = user_report(df, dfnew, region1,country)
    
    
    
    ## Predict using the inputs from the user
    user_data = user_data[X.columns]
    
    threshold = 0.7
    
    user_data["nkill"] = np.log1p(user_data["nkill"])
    user_data["nwound"] = np.log1p(user_data["nwound"])
    # user_result = (classifier.predict_proba(user_data)[0][1] >= threshold).astype(int)
    
    # Obtener probabilidad de clase 'Success'
    proba = classifier.predict_proba(user_data)[0][1]
    user_result = int(proba >= threshold)
    st.markdown(f"🔢 Model confidence for 'Success': **{proba:.2%}**")
    
    # SHAP Explanation
    explainer = shap.Explainer(classifier, X_train)
    shap_values = explainer(user_data)
    
    st.subheader("🔍 SHAP Explanation for This Prediction")
    
    # Convert SHAP values to a summary table
    shap_df = pd.DataFrame({
        "Feature": user_data.columns,
        "SHAP Value": shap_values.values[0],
        "Feature Value": user_data.iloc[0].values
    }).sort_values(by="SHAP Value", key=abs, ascending=False)
    
    st.dataframe(shap_df)
    
    # Descargar explicación SHAP como CSV
    csv = shap_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download SHAP explanation as CSV",
        data=csv,
        file_name="shap_explanation.csv",
        mime='text/csv'
    )
    
    # Crear figura SHAP y pasarla a Streamlit de forma segura
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)
    
    ## Result prediction
    
    
    st.subheader("The algorithm predicts that the terrorist attack would:")
    output = "Succeed" if user_result == 1 else "Failed"
    st.header(output)
    
    proba = classifier.predict_proba(user_data)[0][1]
    st.markdown(f"Model confidence for 'Success': **{proba:.2f}**")

with tab3:
    
    st.markdown("---")
    st.header("🔮 Prospective Attack Risk Simulation")
    
    st.markdown("Simulate a possible future scenario to estimate the historical likelihood of a terrorist attack on a given date and location.")
    
    # --- Inputs del usuario
    future_country = country  # del sidebar
    future_city = st.selectbox("City", df[df["country_txt"] == future_country]["city"].unique())
    future_date = st.date_input("Select a future date", value=pd.to_datetime("2026-12-30"))
    
    
    st.markdown(f"Selected scenario: **{future_city}, {future_country}**")
    geolocator = Nominatim(user_agent="future_scenario_sim")
    location = geolocator.geocode(f"{future_city}, {future_country}")
    
    if location:
        map_df = pd.DataFrame({
            'lat': [location.latitude],
            'lon': [location.longitude]
        })
        st.map(map_df, zoom=6)
    else:
        st.warning("Could not locate the city on the map.")
    
    
    # --- Extracción de día/mes
    day = future_date.day
    month = future_date.month
    
    # --- Filtrado histórico
    df_city = df[(df["country_txt"] == future_country) & (df["city"] == future_city)]
    
    # --- ¿Hubo ataques ese día en la historia?
    matching_dates = df_city[(df_city["iday"] == day) & (df_city["imonth"] == month)]
    
    # --- Estadísticas generales
    attack_rate = round(len(matching_dates) / max(1, len(df_city)) * 100, 2)
    common_type = matching_dates["attacktype1_txt"].mode().iloc[0] if not matching_dates.empty else "N/A"
    avg_kill = round(matching_dates["nkill"].mean(), 1) if not matching_dates.empty else 0
    avg_wound = round(matching_dates["nwound"].mean(), 1) if not matching_dates.empty else 0
    
    # --- Resultados
    st.subheader(f"🗓 Attack risk for {future_city}, {future_country} on {future_date.strftime('%d/%m/%Y')}:")
    st.markdown(f"**📊 Historical probability of attack:** {attack_rate:.2f}%")
    st.markdown(f"**🎯 Most common type of attack:** {common_type}")
    st.markdown(f"**⚰️ Average casualties:** {avg_kill} killed, {avg_wound} wounded")
    
    if matching_dates.empty:
        st.info("ℹ️ No historical attacks found for this specific day. Data may be sparse.")
    
    st.markdown("## 🧠 Simulate This Scenario with the ML Model")
    
    if not df_city.empty:
        if st.button("🧪 Simulate potential attack in model"):
            # 1. Tomar valores históricos más frecuentes
            sample = df_city.dropna().sample(1)  # o usar .mode() para valores más comunes
    
            # 2. Construir input para el modelo
            sim_input = pd.DataFrame({
                "imonth": [month],
                "iday": [day],
                "attacktype1": sample["attacktype1"].values[0],
                "targtype1": sample["targtype1"].values[0],
                "natlty1": sample["natlty1"].values[0],
                "weaptype1": sample["weaptype1"].values[0],
                "region": sample["region"].values[0],
                "latitude": sample["latitude"].values[0],
                "longitude": sample["longitude"].values[0],
                "specificity": sample["specificity"].values[0],
                "vicinity": sample["vicinity"].values[0],
                "extended": sample["extended"].values[0],
                "suicide": sample["suicide"].values[0],
                "nkill": np.log1p(sample["nkill"].values[0]),
                "nwound": np.log1p(sample["nwound"].values[0]),
            })
    
            # 3. Alinear columnas
            sim_input = sim_input[X.columns]
    
            # 4. Hacer predicción
            sim_proba = classifier.predict_proba(sim_input)[0][1]
            sim_result = "Succeed" if sim_proba >= threshold else "Failed"
            
            # Valores simulados (log1p invertido)
            pred_kill = int(np.expm1(sim_input["nkill"].values[0]))
            pred_wound = int(np.expm1(sim_input["nwound"].values[0]))
            
            # Explicación más humana
            st.subheader("🧠 Predicted Outcome")
            st.markdown(
                f"""
                Based on historical patterns and model analysis:
            
                > There is a **{sim_proba:.2%}** probability that an attack in **{future_city}, {future_country}**
                on **{future_date.strftime('%d/%m/%Y')}** would **{sim_result}** if it occurred.
            
                Estimated impact:
                - 🟥 **{pred_kill} fatalities**
                - 🟧 **{pred_wound} injuries**
                """
            )
    
            # 6. SHAP Explanation
            sim_explainer = shap.Explainer(classifier, X_train)
            sim_shap = sim_explainer(sim_input)
    
            st.subheader("🔍 SHAP Explanation for Simulated Scenario")
            sim_df = pd.DataFrame({
                "Feature": sim_input.columns,
                "SHAP Value": sim_shap.values[0],
                "Feature Value": sim_input.iloc[0].values
            }).sort_values(by="SHAP Value", key=abs, ascending=False)
    
            st.dataframe(sim_df)
    
            fig_sim, ax_sim = plt.subplots(figsize=(8, 6))
            shap.plots.waterfall(sim_shap[0], max_display=10, show=False)
            st.pyplot(fig_sim)

with tab4:
    st.subheader("📥 Download Prediction Outputs")

    # Último input del usuario (reconvertido)
    export_input = user_data.copy()
    export_input["nkill"] = np.expm1(export_input["nkill"])
    export_input["nwound"] = np.expm1(export_input["nwound"])
    export_input["Prediction"] = ["Succeed" if user_result == 1 else "Failed"]
    export_input["Confidence"] = [f"{proba:.2%}"]

    input_csv = export_input.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download user input + prediction",
        data=input_csv,
        file_name="prediction_summary.csv",
        mime="text/csv"
    )

    # SHAP explanation CSV (ya calculado antes)
    shap_csv = shap_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download SHAP explanation",
        data=shap_csv,
        file_name="shap_explanation.csv",
        mime="text/csv"
    )