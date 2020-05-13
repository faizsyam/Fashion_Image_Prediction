import plotly.express as px
import plotly
import pandas as pd
import json 
from prediction import df

def get_bar_plots():
    return plot_master,plot_sub,plot_art,plot_gen


def barplot_cat(col,df):
    dfnew = pd.DataFrame(df[col].value_counts().sort_values(ascending=False).reset_index())
    dfnew.rename(columns={'index':col, col:'Amount'},inplace=True)
    fig = px.bar(dfnew, x='Amount', y=col, orientation='h', color=col)
    fig.update_layout(showlegend=False) 
    fig.layout.update()
    fig_json = json.dumps(fig,cls=plotly.utils.PlotlyJSONEncoder)
    return fig_json

plot_master = barplot_cat('masterCategory',df)
plot_sub = barplot_cat('subCategory',df)
plot_art = barplot_cat('articleType',df)
plot_gen = barplot_cat('gender',df)