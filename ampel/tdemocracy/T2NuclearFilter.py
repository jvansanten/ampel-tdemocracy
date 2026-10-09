#!/usr/bin/env python
# File:                ampel/tdemocracy/T2NuclearFilter.py
# License:             BSD-3-Clause
# Author:              jannis.necker@gmail.com
# Date:                29.04.2026
# Last Modified Date:  06.07.2026
# Last Modified By:    jannis.necker@gmail.com

from collections.abc import Sequence
from importlib.metadata import version as pkgversion
from typing import Literal

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.coordinates.angles import angular_separation
from numpy.lib.recfunctions import append_fields
from scipy.stats import chi2
from tdemocracy import __version__ as model_version
from tdemocracy.model import (
    Host,
    MeanPosition,
    NuclearTransientReport,
    Object,
    PhotometricPoint,
    TemplateFluxes,
)

from ampel.abstract.AbsTabulatedT2Unit import AbsTabulatedT2Unit
from ampel.abstract.AbsTiedStateT2Unit import AbsTiedStateT2Unit
from ampel.base.AmpelBaseModel import AmpelBaseModel
from ampel.content.DataPoint import DataPoint
from ampel.content.T1Document import T1Document
from ampel.lasair.LasairAnnotator import LasairAnnotator
from ampel.lsst.t0.DecentVroFilter import RUBIN_ALERT_FLAGS
from ampel.model.StateT2Dependency import StateT2Dependency
from ampel.model.UnitModel import UnitModel
from ampel.struct.UnitResult import UnitResult
from ampel.tdemocracy.util.catalog_column_info import (
    get_catalog_position_unit_map,
    get_type_and_redshift_columns,
)
from ampel.types import StockId, UBson
from ampel.view.T2DocView import T2DocView


class NuclearFilterResult(AmpelBaseModel):
    passed: bool
    report: NuclearTransientReport


class T2NuclearFilter(AbsTiedStateT2Unit, AbsTabulatedT2Unit, LasairAnnotator):
    match_dist_arcsec: float
    group_matches_within_arcsec: float = 0.5
    min_reliability: float = 0.8

    t2_dependency: Sequence[
        StateT2Dependency[
            Literal["T2CatalogMatch", "T2LSPhotoZTap", "T2DigestRedshifts"]
        ]
    ]

    tabulator: Sequence[UnitModel] = [UnitModel(unit="LSSTT2Tabulator")]

    result_adapter: UnitModel | None = None
    do_lasair_annotation: bool = True
    always_return_body: bool = True
    lasair_topic: Literal["tdemocracy-nuclear-stream"] = "tdemocracy-nuclear-stream"  # type: ignore
    lasair_version = "lsst"

    version = str(pkgversion("ampel-tdemocracy"))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # set up unit mapping for positions
        unit_mapping = get_catalog_position_unit_map()
        convert_to_rad = []
        for name, units in unit_mapping.items():
            if ((u := units["ra"]["unit"]) is None) or (u.lower().startswith("deg")):
                convert_to_rad.append(name)
        self._known_mapping = [*list(unit_mapping.keys()), "T2LSPhotoZTap"]
        self._convert_to_rad = [*convert_to_rad, "T2LSPhotoZTap"]
        self._redshift_columns, self._type_columns = get_type_and_redshift_columns()
        self._percentile_2dsig = chi2.cdf(1, 2)

    def _get_photometric_points(
        self, datapoints: Sequence[DataPoint]
    ) -> list[PhotometricPoint]:
        return [
            PhotometricPoint(
                id=int(id),
                source=source,
                time=time,
                flux=flux,
                fluxerr=fluxerr,
                band=band,
                zp=zp,
                zpsys=zpsys,
            )
            # NB: this assumes that the flux table includes id and source columns, which only LSSTT2Tabulator does. If other tabulators are used, this may need to be adjusted.
            for id, source, time, flux, fluxerr, band, zp, zpsys in self.get_flux_table(
                datapoints
            ).iterrows("id", "source", "time", "flux", "fluxerr", "band", "zp", "zpsys")
        ]

    def _get_object(
        self, datapoints: Sequence[DataPoint], stock: StockId | Sequence[StockId]
    ) -> Object:
        # Fill object record from latest LSST_OBJ datapoint (diaObject)
        for dp in sorted(
            datapoints, key=lambda x: x["meta"][-1].get("ts", 0), reverse=True
        ):
            if "LSST_OBJ" in dp.get("tag", {}):
                if isinstance(stock, Sequence):
                    raise NotImplementedError("Can not handle multiple stock IDs yet!")
                try:
                    stock_int = int(stock)
                except ValueError as e:
                    raise ValueError(f"Did not understand Rubin ID {stock}!") from e

                return Object(
                    id=stock_int,
                    source="LSST",
                    ra=float(dp["body"]["ra"]),
                    ra_err=float(dp["body"]["raErr"]),
                    dec=float(dp["body"]["dec"]),
                    dec_err=float(dp["body"]["decErr"]),
                    ra_dec_cov=float(dp["body"]["ra_dec_Cov"]),
                    # FIXME: add redshift if available
                )
            # TODO: add ZTF ID eventually if ZTF matcheing data becomes available
            # if "ZTF" in dp.get("tag", {}) and "ra" in dp["body"]:
            #     return Object(
            #         id=stock,
            #         external_id=ZTFIdMapper.to_ext_id(stock),
            #         source="ZTF",
            #         ra=float(dp["body"]["ra"]),
            #         dec=float(dp["body"]["dec"]),
            #     )
        raise ValueError("No Object found in datapoints")

    @staticmethod
    def _get_template_fluxes(datapoints: Sequence[DataPoint]) -> TemplateFluxes:
        # calculate template fluxes
        _t = {}
        for b in "ugrizy":
            band_dps = [
                dp["body"]["templateFlux"]
                for dp in datapoints
                if ("templateFlux" in dp) and (dp["body"]["band"] == b)
            ]
            if any(band_dps):
                median, perc5, perc95 = np.quantile(
                    band_dps, [0.5, 0.05, 0.95]
                ).tolist()
                _t[b] = {"band": b, "median": median, "perc5": perc5, "perc95": perc95}
            else:
                _t[b] = None  # type: ignore
        return TemplateFluxes(**_t)

    def process(
        self,
        compound: T1Document,
        datapoints: Sequence[DataPoint],
        t2_views: Sequence[T2DocView],
    ) -> UBson | UnitResult:
        md = self.match_dist_arcsec

        # Collect all matches from the match units that are within self.match_dist_arcsec
        # Attention! If there are multiple catalog match units looking at the same catalogs
        # with different configurations the matches will be overwritten!
        matches = {}
        digest_redshifts = None
        for t2_view in t2_views:
            if (t2_view.unit in {"T2CatalogMatch", "T2LSPhotoZTap"}) and (
                body := t2_view.get_payload()
            ) is not None:
                matches.update(
                    {
                        k: v
                        for k, v in body.items()
                        if (v is not None) and (v["dist2transient"] <= md)
                    }
                )
            if t2_view.unit == "T2DigestRedshifts":
                digest_redshifts = t2_view.get_payload()

        if not digest_redshifts:
            raise RuntimeError("Missing T2 dependencies: T2DigestRedshifts!")

        # --- select reliable data --- #

        # exclude flagged visits or in period with duplicated data
        bad_visits = set(
            visit
            for dp in datapoints
            if ((visit := dp["body"].get("visit")) is not None)
            and (
                any([dp["body"].get(flag) for flag in RUBIN_ALERT_FLAGS])
                or ((visit > 2026021900000) and (visit < 2026022400000))
            )
        )
        good_sources = [
            dp
            for dp in datapoints
            if ("diaSourceId" in dp["body"])
            and (dp["body"]["visit"] not in bad_visits)
            and dp["body"]["reliability"] >= self.min_reliability
        ]
        good_forced_sources = [
            dp
            for dp in datapoints
            if ("diaForcedSourceId" in dp["body"])
            and (dp["body"]["visit"] not in bad_visits)
        ]
        good_datapoints = good_sources + good_forced_sources

        # --- calculate mean position and variance --- #

        coords = SkyCoord(
            *np.array(
                [[dp["body"][k] for k in ["ra", "dec"]] for dp in good_sources]
            ).T,
            unit="deg",
        )

        circularized_errors = np.array(
            [
                np.sqrt(dp["body"]["raErr"] ** 2 + dp["body"]["decErr"] ** 2)
                for dp in good_sources
            ]
        )
        weights = 1 / circularized_errors
        normed_weights = weights / np.sum(weights)
        mean_pos = SkyCoord((coords.represent_as("cartesian") * normed_weights).sum())
        mean_pos.representation_type = "spherical"

        circularized_mean_error = (
            np.sqrt(sum((normed_weights * circularized_errors) ** 2)) * 3600
        )
        separations_to_mean = coords.separation(mean_pos).to_value("arcsec")
        std = np.std(separations_to_mean, ddof=1)

        mean_ra = mean_pos.ra.to_value("deg")
        mean_dec = mean_pos.dec.to_value("deg")

        report = NuclearTransientReport(
            photometry=self._get_photometric_points(good_datapoints),
            object=self._get_object(datapoints, compound["stock"]),
            template_fluxes=self._get_template_fluxes(good_datapoints),
            version=self.version,
            model_version=model_version,
            state=compound["link"],
            mean_position=MeanPosition(
                mean_ra=mean_ra,
                mean_dec=mean_dec,
                std=float(std),
                circularized_error=float(circularized_mean_error),
                n_sources=len(good_sources),
            ),
            host=None,
        )

        # If there are no matches within self.match_dist_arcsec, return False
        if not matches:
            return UnitResult(
                body=NuclearFilterResult(
                    passed=False,
                    report=report,
                ).model_dump()
            )

        # normalize keys
        matches = {
            k: {kk.lower(): vv for kk, vv in v.items()} for k, v in matches.items()
        }

        # PS1_photoz is special
        if "PS1_photoz" in matches:
            matches["PS1_photoz"]["ra"] = matches["PS1_photoz"]["ramean"]
            matches["PS1_photoz"]["dec"] = matches["PS1_photoz"]["decmean"]

        # remove matches with unknown units
        for m in [mm for mm in matches if mm not in self._known_mapping]:
            self.logger.info(f"Removing {m} from matches because unit is unknown")
            matches.pop(m)

        # convert to radians where necessary
        for name in matches:
            for k in ["ra", "dec"]:
                if k not in matches[name]:
                    raise KeyError(f"Key {k} not found in matches")
                if name in self._convert_to_rad:
                    matches[name][k] = np.radians(float(matches[name][k]))
                else:
                    matches[name][k] = float(matches[name][k])

        # find the closest match
        try:
            match_map = np.array(
                [(k, float(v["ra"]), float(v["dec"])) for k, v in matches.items()],
                dtype=[
                    ("name", "<U30"),
                    ("ra_rad", "<f8"),
                    ("dec_rad", "<f8"),
                ],
            )

        except KeyError as e:
            raise e

        distance_to_mean_pos = mean_pos.separation(
            SkyCoord(match_map["ra_rad"], match_map["dec_rad"], unit="rad")
        ).to_value("arcsec")
        match_map_with_dist = append_fields(
            match_map,  # Original structured array
            "distance",  # Name(s) of new field(s) (string or list of strings)
            distance_to_mean_pos,  # Data for new field(s) (array-like or list of array-likes)
        )

        best_match_id = np.argmin(match_map_with_dist["distance"])
        dist = match_map_with_dist["distance"][best_match_id]

        # find matches that are probably the same object
        separations = (
            np.degrees(
                angular_separation(
                    match_map_with_dist["ra_rad"][best_match_id],
                    match_map_with_dist["dec_rad"][best_match_id],
                    match_map_with_dist["ra_rad"],
                    match_map_with_dist["dec_rad"],
                )
            )
            * 3600
        )
        matched_catalogs = match_map_with_dist["name"][
            separations <= self.group_matches_within_arcsec
        ]
        passed = bool(dist <= md)

        type_info = {
            k: {kk: v[kk] for kk in self._type_columns if kk in v}
            for k, v in matches.items()
        }

        if ("T2LSPhotoZTap" in type_info) and (
            type_info["T2LSPhotoZTap"]["type"] in {"DUP", "PSF"}
        ):
            passed = False

        report.host = Host(
            name="T2NuclearFilter",
            primary_source=match_map_with_dist["name"][best_match_id],
            ra=float(np.degrees(match_map_with_dist["ra_rad"][best_match_id])),
            dec=float(np.degrees(match_map_with_dist["dec_rad"][best_match_id])),
            redshift=digest_redshifts.get("ampel_z"),
            distance=dist,
            sources=matched_catalogs.tolist(),
            info=type_info,
        )
        assert report.host is not None  # this is just for mypy

        if passed:
            if self.do_lasair_annotation and passed:
                annotation_succeeded = self.annotate(
                    str(report.object.id),
                    classification="nuclear",
                    version=self.version,
                    explanation=f"Extended host in LS DR10 within {self.match_dist_arcsec} arcsec",
                    classdict={
                        "host_distance": report.host.distance,
                        "mean_position_ra": report.mean_position.mean_ra,
                        "mean_position_dec": report.mean_position.mean_dec,
                        "mean_position_std": report.mean_position.std,
                    },
                )
                if not annotation_succeeded:
                    self.logger.error(
                        f"Lasair annotation failed for {report.object.id}!"
                    )
            else:
                self.logger.info(
                    f"Skipping Lasair annotation for {report.object.id} as requested"
                )

        result = NuclearFilterResult(passed=passed, report=report)
        return UnitResult(body=result.model_dump(), adapter=self.result_adapter) if passed or self.always_return_body else None
