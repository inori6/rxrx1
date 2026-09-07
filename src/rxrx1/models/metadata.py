import torch
import torch.nn as nn
import torch.nn.functional as F


class MetadataFusion(nn.Module):
    def __init__(
        self,
        feature_dim,
        method="concat",
        cell_type=False,
        well_position=False,
        num_cell_types=4,
        well_dim=2,
    ):
        super().__init__()

        if method not in {"concat", "film"}:
            raise ValueError(f"Unsupported metadata method: {method}")
        if not cell_type and not well_position:
            raise ValueError("Enable at least one metadata type")

        self.method = method
        self.use_cell_type = cell_type
        self.use_well_position = well_position
        self.num_cell_types = num_cell_types

        self.meta_dim = (
            num_cell_types * int(cell_type)
            + well_dim * int(well_position)
        )

        if method == "film":
            if cell_type:
                self.cell_film = nn.Linear(
                    num_cell_types,
                    feature_dim * 2,
                )
                nn.init.zeros_(self.cell_film.weight)
                nn.init.zeros_(self.cell_film.bias)

            if well_position:
                self.well_film = nn.Linear(
                    well_dim,
                    feature_dim * 2,
                )
                nn.init.zeros_(self.well_film.weight)
                nn.init.zeros_(self.well_film.bias)

        self.out_dim = (
            feature_dim + self.meta_dim
            if method == "concat"
            else feature_dim
        )

    def forward(self, x, metadata):
        concat_metadata = []

        if self.use_cell_type:
            cell = F.one_hot(
                metadata["cell_type_idx"].long(),
                num_classes=self.num_cell_types,
            ).float()

            if self.method == "concat":
                concat_metadata.append(cell)
            else:
                gamma, beta = self.cell_film(cell).chunk(2, dim=1)
                shape = (*gamma.shape, *((1,) * (x.ndim - 2)))
                gamma, beta = gamma.reshape(shape), beta.reshape(shape)
                x = (1 + gamma) * x + beta

        if self.use_well_position:
            well = metadata["well_position"].float()

            if self.method == "concat":
                concat_metadata.append(well)
            else:
                gamma, beta = self.well_film(well).chunk(2, dim=1)
                shape = (*gamma.shape, *((1,) * (x.ndim - 2)))
                gamma, beta = gamma.reshape(shape), beta.reshape(shape)
                x = (1 + gamma) * x + beta

        if self.method == "concat":
            x = torch.cat([x, *concat_metadata], dim=1)

        return x


if __name__ == "__main__":
    x = torch.randn(8, 1408)

    metadata = {
        "cell_type_idx": torch.randint(0, 4, (8,)),
        "well_position": torch.rand(8, 2),
    }

    concat = MetadataFusion(
        feature_dim=1408,
        method="concat",
        cell_type=True,
        well_position=True,
    )
    print("concat:", concat(x, metadata).shape)

    film = MetadataFusion(
        feature_dim=1408,
        method="film",
        cell_type=True,
        well_position=True,
    )
    print("film:", film(x, metadata).shape)

    film_output = film(x, metadata)
    print("film_output:", film_output)
    print(torch.allclose(x, film_output))
