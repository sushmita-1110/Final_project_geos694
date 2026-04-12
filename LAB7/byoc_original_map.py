import pandas as pd
import pygmt

df = pd.read_csv("gmap-stations-AK.txt", sep="|")
df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
df = df.dropna(subset=["Latitude", "Longitude"])

region = [-170, -135, 51, 70]
projection = "L-150/62/55/65/12c"
grid = "@earth_relief_03m"

fig = pygmt.Figure()
fig.grdimage(grid=grid, region=region, projection=projection, shading=True, cmap="geo")
fig.coast(
    region=region,
    projection=projection,
    shorelines="0.6p,black",
    borders=["1/0.5p,black", "2/0.25p,gray40"],
    rivers="a/0.25p,blue",
    lakes="lightblue",
    resolution="i",
)
fig.plot(
    x=df["Longitude"],
    y=df["Latitude"],
    style="i0.19c",
    fill="red",
    pen="0.25p,black",
)
fig.text(
    x=df["Longitude"],
    y=df["Latitude"],
    text=df["Station"],
    font="1.9p,Helvetica-Bold,black",
    justify="LT",
    offset="0.05c/0.05c",
    fill="white",
)
fig.basemap(frame=['af', '+tAlaska Stations'])
fig.savefig("AK_station_ALASKA_map.png", dpi=800)
fig.show()