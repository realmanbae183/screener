import sys, io
_buf=io.StringIO(); _o=sys.stdout; sys.stdout=_buf
import yfinance as yf, pandas as pd
for sym in ["TQQQ","QQQ","NQ=F"]:
    df=yf.download(sym,period="2d",interval="1m",prepost=True,progress=False,auto_adjust=False)
    if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
    df.index=pd.to_datetime(df.index,utc=True).tz_convert("America/New_York")
    d=df.index[-1].date(); day=df[df.index.date==d]
    reg=day.between_time("09:30","16:00"); orb=day.between_time("09:30","09:59")
    print(sym,d,"last",df.index[-1].strftime("%H:%M"),round(float(df.Close.iloc[-1]),2))
    if len(reg): print("  open",round(float(reg.Open.iloc[0]),2),"ORhi",round(float(orb.High.max()),2),"ORlo",round(float(orb.Low.min()),2))
    for hh in ("10:00","11:00","12:00","13:00"):
        p=day[day.index<=pd.Timestamp(f"{d} {hh}",tz="America/New_York")]
        if len(p) and p.index[-1].strftime("%H:%M")>="09:30": print("  ",hh,round(float(p.Close.iloc[-1]),2))

sys.stdout=_o; open('lab/results/today.txt','w').write(_buf.getvalue()); print(_buf.getvalue())
