import folium
import branca.colormap as cm

class BuildMap:

    @staticmethod
    def build_map(points_list, connections_list, colors_list):

        # Create a map centered around Barcelona
        map_barcelona = folium.Map(location=[41.3851, 2.1734], zoom_start=13, tiles="cartodb positron")

        # Add markers for each point of interest
        i = 0
        for name, coord in points_list.items():
            folium.Marker(
                location=coord,
                popup=name,
                icon=folium.Icon(color=colors_list[i], icon="circle")
            ).add_to(map_barcelona)
            i += 1

        # Add lines connecting the points
        for i in range(len(connections_list[0])):
            folium.PolyLine(
                connections_list[0][i],
                color=connections_list[1][i],
                weight=2.5,
                opacity=1
            ).add_to(map_barcelona)

        # Display the map
        map_barcelona.save("barcelona_map.html")

    @staticmethod
    def build_map(points_list, connections_list, output_name):
        # Create a map centered around Barcelona
        # map_barcelona = folium.Map(location=[41.3851, 2.1734], zoom_start=12, tiles="cartodb positron")
        map_barcelona = folium.Map(location=[41.3950, 2.1734], zoom_start=12, tiles="cartodb positron")


        # Add markers for each point of interest
        i = 0
        for name, coord in points_list.items():
            # folium.Marker(
            #     location=coord,
            #     popup=folium.Popup(name),
            #     # icon=folium.Icon(color="red")
            #     icon=folium.DivIcon(html=f"""<h3 style="size=5; color: red">{name}</div>""")
            # ).add_to(map_barcelona)
            folium.CircleMarker(
                [coord[0], coord[1]], 
                radius=10, 
                color="red", 
                fill=True, 
                fill_color="red"
            ).add_to(map_barcelona)
            i += 1

        # Add lines connecting the points
        for i in range(len(connections_list[0])):
            folium.PolyLine(
                connections_list[0][i],
                color=connections_list[1][i],
                weight=2.5,
                opacity=1
            ).add_to(map_barcelona)

        colormap = cm.LinearColormap(colors=['yellow', 'gold', 'orange', 'red', 'black'],
                             index=[0.2, 0.4, 0.6, 0.8, 1], vmin=0, vmax=1,
                             caption='Level of Connectivity')


        map_barcelona.add_child(colormap)

        output_name = output_name + ".html"

        # Display the map
        map_barcelona.save(output_name)
        map_barcelona = None

