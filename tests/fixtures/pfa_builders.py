# Create pfa dictionaries for testing


def make_pfa_with_layers(layer_gdfs):
    """
    layer_gdfs: dict of {layer_name: GeoDataFrame}
    """
    return {
        "criteria": {
            "crit1": {
                "components": {
                    "comp1": {
                        "layers": {
                            name: {"model": gdf}
                            for name, gdf in layer_gdfs.items()
                        }
                    }
                }
            }
        }
    }
