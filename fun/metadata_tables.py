# Remarks:
# TODO: modify the datatables to match the ISO keywords types:https://wiki.esipfed.org/Documenting_Keywords [Future v2.0]
# TODO: add def __init__(self, parameters...) in each class refer: https://stackoverflow.com/questions/2047814/is-it-possible-to-store-python-class-objects-in-sqlite
# TODO: convert to annotated style

from __future__ import annotations
from typing import List
from sqlalchemy import (Table, Column, Integer, String, ForeignKey, MetaData, insert, create_engine, Enum, \
            Float, select, Boolean, DateTime, Interval, ARRAY, )
from sqlalchemy.orm import (relationship, declarative_base, registry, scoped_session, sessionmaker, Session, backref, mapped_column, MappedAsDataclass, DeclarativeBase)
from sqlalchemy.ext.serializer import loads, dumps

import enum
import uuid
import os, sys
import datetime
import numpy as np
sys.path.append(r"../fun")
import pythonAssist as pa
import pandas as pd
from datetime import datetime, timedelta
from enumerations import *
from metadata_helpers import *
from ArrayType import ArrayType
import isodate

class Base(DeclarativeBase):
    """subclasses will be converted to dataclasses"""

class User(Base):
    __tablename__ = "user"

    id = mapped_column( Integer, primary_key = True,autoincrement = True)
    name = mapped_column(
        Enum(peopleEnum),
        default="ashim",
        comment="name of the user")
    url = mapped_column(
        String(),
        default=
        "https://fraunhofer.sharepoint.com/sites/IWES-OE440/SitePages/en/OE-443.aspx",
        comment="data management institution webpage",
        nullable=True,
    )
    email = mapped_column(
        String(),
        default="ashim.giyanani@iwes.fraunhofer.de",
        comment="email address of the user",
        nullable=True,
    )
    tel = mapped_column(
        String(),
        default="+49 151 42462025",
        comment="telephone contact of the user",
        nullable=True,
    )
    institution = mapped_column(
        String(),
        default="Fraunhofer IWES",
        comment="name of the company/institution",
        nullable=True,
    )
    address = mapped_column(
        String(),
        default='Am Seedeich 45, 27572 Bremerhaven, Deutschland',
        comment="address of the user",
        nullable=True)
    # ds_infos = relationship("DatasetInfo", back_populates="users", lazy='select')


class DatasetInfo(Base):
    __tablename__ = "dataset_info"
    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    webpage = mapped_column(
        Enum(webpageEnum),
        # ForeignKey("user_interface.system_name"),
        default="onedas",
        comment="webpage for general description and access",
        nullable=True,
    )
    url = mapped_column(
        Enum(UrlEnum),
        # ForeignKey("user_interface.url"),
        default="https://onedas.iwes.fraunhofer.de",
        nullable=True,
        comment="url of the webpage for access")
    name = mapped_column(
        String(),
        default="testfeld",
        comment="name of the dataset collection",
        nullable=False,
    )
    alias = mapped_column(
        String(),
        default=None,
        comment="Alias or pseudo name for the dataset",
        nullable=True,
    )
    authority = mapped_column(
        String(),
        default="Fraunhofer IWES",
        comment="Authority responsible for the dataset",
        nullable=True,
    )
    collection_type = mapped_column(
        Enum(cdmEnum),
        default="ts",
        comment="The type of dataset collection",
        nullable=True,
    )
    # data_type = mapped_column(
    #     String(),
    #     default = "point",
    #     comment = "high-level semantic type of dataset (e.g., grid, point, trajectory) \
    #                 and can be used by clients to decide how to display the data"
    # )
    acknowledgement = mapped_column(
        String(),
        default=None,
        comment=
        "A place to acknowledge various types of support for the project that produced this data",
        nullable=True,
    )
    cdm_data_type = mapped_column(
        Enum(cdmEnum),
        default='ts',
        comment=
        "The data type, as derived from Unidata's Common Data Model Scientific Data types and understood by THREDDS. (This is a THREDDS dataType, and is different from the CF NetCDF attribute 'featureType', which indicates a Discrete Sampling Geometry file in CF.)",
        nullable=False,
    )
    convention = mapped_column(
        String(),
        default="CF-1.7/ACDD-1.3",
        comment="States that the CF convention is being used and what version",
        nullable=False,
    )
    coverage_content_type = mapped_column(
        Enum(contentEnum),
        default="physical",
        comment=
        "An ISO 19115-1 code to indicate the source of the data (image, thematicClassification, physicalMeasurement, auxiliaryInformation, qualityInformation, referenceInformation, modelResult, or coordinate).",
        nullable=False,
    )
    creator_id = mapped_column(Integer,
                               # ForeignKey('user.id'),
                               )
    creator_name = mapped_column(
        String,
        # ForeignKey('user.name'),
        comment="name of the data manager",
        nullable=True,
    )
    creator_url = mapped_column(
        String,
        # ForeignKey('user.url'),
        comment="creator's institution webpage",
        nullable=True,
    )
    creator_email = mapped_column(
        String,
        # ForeignKey('user.email'),
        comment="email address of the data manager",
        nullable=True,
    )
    creator_tel = mapped_column(
        String,
        # ForeignKey('user.tel'),
        comment="telephone contact of the data manager",
        nullable=True,
    )
    creator_institution = mapped_column(
        String,
        # ForeignKey('user.institution'),
        comment="name of the company/institution",
        nullable=True,
    )
    subject = mapped_column(
        Enum(subjectEnum),
        default="wind energy lidar measurements of the atmosphere",
        comment=
        "A topic of the resource, typically keywords, key phrases or classification codes",
        nullable=True,
    )
    description = mapped_column(
        String(),
        default=
        "Example: 10 minute average data from romoWind's iSpin sonic anemometer installed on the hub of a wind turbine at Testfeld BHV, Bremerhaven for Fraunhofer IWES within the framework of Testfeld project, 24.07.2019 to 24.07.2022",
        comment=
        "description of the file answering 5W's (Who, What, Where, Why, When)",
        nullable=True,
    )
    institution = mapped_column(
        String(),
        default="Fraunhofer IWES",
        comment="Specifies where the original data was produced",
        nullable=False,
    )
    references = mapped_column(
        String(),
        default=None,
        comment=
        "Published or web-based references that describe the data or methods used to produce it",
        nullable=True,
    )
    source = mapped_column(
        Enum(DataSourceEnum),
        default="surface",
        comment=
        "The method of production of the original data. If it was model-generated, source should name the model and its version, as specifically as could be useful. If it is observational, source should characterize it (e.g., surface observation or radiosonde)",
        nullable=False,
    )
    history = mapped_column(
        String(),
        default=None,
        comment=
        "Provides an audit trail for modifications to the original data. Well-behaved generic netCDF filters will automatically append their name and the parameters with which they were invoked to the global history attribute of an input netCDF file. We recommend that each line begin with a timestamp indicating the date and time of day that the program was executed",
        nullable=True,
    )
    comment = mapped_column(
        String(),
        default=None,
        comment=
        "Miscellaneous information about the data or methods used to produce it.",
        nullable=True,
    )
    publisher_name = mapped_column(
        Enum(peopleEnum),
        # ForeignKey('user.name'),
        comment="name of the person responsible for metadata and formatting",
        nullable=True,
    )
    publisher_email = mapped_column(
        String(),
        # ForeignKey('user.email'),
        comment="contact person's email",
        nullable=True,
    )
    publisher_tel = mapped_column(
        String(),
        # ForeignKey('user.tel'),
        comment="contact person's telephone Nr.",
        nullable=True,
    )
    publisher_institution = mapped_column(
        String(),
        # ForeignKey('user.institution'),
        comment="Institute's name",
        nullable=True,
    )
    publisher_url = mapped_column(
        String(),
        # ForeignKey('user.url'),
        comment="Website of the institute",
        nullable=True,
    )
    publish_uuid = mapped_column(
        String(),
        default=None,
        comment="unique id of the data publication",
        nullable=True,
    )
    doi = mapped_column(
        String(),
        default=None,
        comment="DOI of the publication",
        nullable=True,
    )
    handle = mapped_column(
        String(),
        default=None,
        comment="mostly in case of git repositories or online",
        nullable=True,
    )
    product_version = mapped_column(
        String(),
        default=None,
        comment="Version identifier of the data file ",
        nullable=True,
    )
    contributor_name = mapped_column(
        Enum(peopleEnum),
        default=None,
        comment=
        "The name of any individuals, projects, or institutions that contributed to the creation of this data. May be presented as free text, or in a structured format compatible with conversion to ncML (e.g., insensitive to changes in whitespace, including end-of-line characters)",
        nullable=True,
    )
    contributor_role = mapped_column(
        String(),
        default=None,
        comment=
        "The role of any individuals, projects, or institutions that contributed to the creation of this data. May be presented as free text, or in a structured format compatible with conversion to ncML (e.g., insensitive to changes in whitespace, including end-of-line characters). Multiple roles should be presented in the same order and number as the names in contributor_names",
        nullable=True,
    )
    type = mapped_column(
        String(),
        default=None,
        comment="The nature or genre of the resource.",
        nullable=True,
    )
    format = mapped_column(
        Enum(FileFormatEnum),
        default="ascii",
        comment=
        "The file format, physical medium, or dimensions of the resource",
        nullable=False,
    )
    language = mapped_column(
        String(),
        default="en",
        comment="language of metadata",
        nullable=False,
    )
    relation = mapped_column(
        String(),
        default=None,
        comment="A related resource identified using a URI",
        nullable=True,
    )
    # users = relationship("User", lazy='select', back_populates='dataset_info')

    def __repr__(self):
        return "DatasetInfo(webpage = '{self.webpage}', " \
            "url  = '{self.url}',  " \
            "name  = '{self.name}',  " \
            "alias  = '{self.alias}',  "\
            "authority  = '{self.authority}',  "\
            "collection_type  = '{self.collection_type}',  "\
            "acknowledgement  = '{self.acknowledgement}',  "\
            "cdm_data_type  = '{self.cdm_data_type}',  "\
            "convention  = '{self.convention}',  "\
            "coverage_content_type  = '{self.coverage_content_type}',  "\
            "creator_name  = '{self.creator_name}',  "\
            "creator_url  = '{self.creator_url}',  "\
            "creator_email  = '{self.creator_email}',  "\
            "creator_tel  = '{self.creator_tel}',  "\
            "creator_institution  = '{self.creator_institution}',  "\
            "subject  = '{self.subject}',  "\
            "description  = '{self.description}',  "\
            "institution  = '{self.institution}',  "\
            "references  = '{self.references}',  "\
            "source  = '{self.source}',  "\
            "history  = '{self.history}',  "\
            "comment  = '{self.comment}',  "\
            "publisher_name  = '{self.publisher_name}',  "\
            "publisher_email  = '{self.publisher_email}',  "\
            "publisher_tel  = '{self.publisher_tel}',  "\
            "publisher_institution  = '{self.publisher_institution}',  "\
            "publisher_url  = '{self.publisher_url}',  "\
            "publish_uuid  = '{self.publish_uuid}',  "\
            "doi  = '{self.doi}',  "\
            "handle  = '{self.handle}',  "\
            "product_version  = '{self.product_version}',  "\
            "contributor_name  = '{self.contributor_name}',  "\
            "contributor_role  = '{self.contributor_role}',  "\
            "type  = '{self.type}',  "\
            "format  = '{self.format}',  "\
            "language  = '{self.language}',  "\
            "relation  = '{self.relation}')".format(self=self)


class UserInterface(Base):
    __tablename__ = "user_interface"
    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    project_name = mapped_column(
        Enum(projectEnum),
        # default="highre",
        comment = "name of the project",
        primary_key=False,
        nullable=False,
        )
    system_name = mapped_column(
        Enum(SystemEnum),
        default="onedas",
        comment= "name of the GUI or system where the data is accessible",
        nullable=False,
    )
    url = mapped_column(
        Enum(UrlEnum),
        default="onedas",
        nullable = False,
        comment="url of the webpage for access",
    )
    filename = mapped_column(
        String(),
        default = None,
        comment = "name of the file where the data is stored",
        nullable=True,
    )
    extension = mapped_column(
        Enum(FileFormatEnum),
        default = "ascii",
        comment = "file extension",
        nullable=True,
    )
    access = mapped_column(
        Enum(AccessEnum),
        default="limited access",
        comment = "Extent to which the data is accessible",
        nullable=True,
    )
    start_datetime = mapped_column(
        DateTime(),
        default = pa.now(),
        comment = "start datetime of the file extracted from oneDas",
        onupdate=pa.now(),
        nullable=True,
    )
    end_datetime = mapped_column(
        DateTime(),
        default = pa.now(),
        comment = "end datetime of the file extracted from oneDas",
        onupdate=pa.now(),
        nullable=True,
    )
    file_granularity = mapped_column(
        Enum(FileGranularityEnum),
        default = "1 file/day",
        comment = "duration of the file extracted from oneDas or other other system",
        nullable=False,
    )
    sample_rate = mapped_column(
        String(),
        default = "1 Hz",
        # units = "Hz",
        comment = "sampling frequency of the measurement",
        nullable=False,
    )
    file_format_version = mapped_column(
        String(),
        default = "1.0",
        comment = "file format version",
        nullable=True,
    )
    files_per_day = mapped_column(
        Integer,
        default = 144,
        comment = "maximum number of files in a day considering the file granularity and 100% availability",
        nullable=True,
    )
    samples_per_file = mapped_column(
        Integer,
        default = 144,
        comment = "number of samples in a day considering the file granularity and sampling rate",
        nullable=True,
    )
    project_id = mapped_column(
        String(),
        default = "/AIRPORT/AD8_PROTOTYPE/GENERAL_DAQ",
        comment = "project id and metadata organization",
        nullable=True,
    )
    license = mapped_column(
        Enum(licenseEnum),
        default = "confidential",
        comment = "data usage agreement and policy",
        nullable=False,
    )
    # dsinfos = relationship("DatasetInfo", back_populates="uis")
    def __repr__(self):
        return "UserInterface(project_name = '{self.project_name}', "\
            "system_name = '{self.system_name}', "\
            "url = '{self.url}', "\
            "filename = '{self.filename}', "\
            "extension = '{self.system_name}', "\
            "access = '{self.access}', "\
            "start_datetime = '{self.start_datetime}', "\
            "end_datetime = '{self.end_datetime}', "\
            "file_granularity = '{self.file_granularity}', "\
            "sample_rate = '{self.sample_rate}', "\
            "file_format_version = '{self.file_format_version}', "\
            "files_per_day = '{self.files_per_day}', "\
            "samples_per_file = '{self.samples_per_file}', "\
            "project_id = '{self.project_id}', "\
            "license = '{self.license}')".format(self.self)

class Network(Base):
    __tablename__ = "network"
    id = mapped_column(Integer,primary_key=True, autoincrement=True)
    ntp_sync = mapped_column(
        String(),
        comment = "time synchronization with NTP (s)",
        # units = "seconds",
        default = (str(timedelta(seconds=10).total_seconds()) + " s"),
        nullable=True,
    )
    ntp_sync_err = mapped_column(
        String(),
        comment = "time synchronization error with NTP (s)",
        # units = "seconds",
        default = (str(timedelta(seconds=0.1).total_seconds()) + " s"),
        nullable=True,
    )
    daylight_saving = mapped_column(
        Boolean,
        default = False,
        comment = "daylight saving applied in the form",
        nullable=True,
    )
    ntp_reftime = mapped_column(
        String(),
        default = "UTC",
        comment = "reference time on the NTP server",
        nullable=False,
    )
    ntp_refserver = mapped_column(
        String(),
        default = "pool.ntp.org",
        comment = "link / details of the NTP reference server",
        nullable=False,
    )
    communication_type = mapped_column(
        Enum(CommunicationEnum),
        default = "broadband",
        comment = "Network communication type (WIFI, broadband, dial-up, LAN,)",
        nullable=False,
    )
    data_trans_freq = mapped_column(
        String(),
        default = "hourly",
        comment = "Data transmission frequency",
        nullable=True,
    )

    def __repr__(self):
        return "Network(ntp_sync = '{self.ntp_sync}', "\
            "ntp_sync_error = '{self.ntp_sync_error}', "\
            "daylight_saving = '{self.daylight_saving}', "\
            "ntp_reftime = '{self.ntp_reftime}', "\
            "ntp_refserver = '{self.ntp_refserver}', "\
            "communication_type = '{self.communication_type}', "\
            "data_transfer_freq = '{self.data_transfer_freq}')".format(self.self)


class DataFile(Base):
    __tablename__ = "datafile"
    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    date_created = mapped_column(
        DateTime(),
        default = pd.to_datetime(pa.now(), utc=True, dayfirst=True),
        comment = "date and time of file creation",
        onupdate=pd.to_datetime(pa.now(), utc=True, dayfirst=True),
        nullable=True,
    )
    date_issued = mapped_column(
        DateTime(),
        default = pd.to_datetime(pa.now(), utc=True, dayfirst=True),
        comment= "The date on which this data (including all modifications) was formally issued",
        onupdate=pd.to_datetime(pa.now(), utc=True, dayfirst=True),
        nullable=True,
        )
    date_modified = mapped_column(
        DateTime(),
        default = pd.to_datetime(pa.now(), utc=True, dayfirst=True),
        comment= "The date on which the data was last modified",
        onupdate=pd.to_datetime(pa.now(), utc=True, dayfirst=True),
        nullable=True,
        )
    date_metadata_modified = mapped_column(
        DateTime(),
        default = pd.to_datetime(pa.now(), utc=True, dayfirst=True),
        comment= "The date on which the metadata was last modified",
        onupdate=pd.to_datetime(pa.now(), utc=True, dayfirst=True),
        nullable=True,
        )
    time_coverage_start = mapped_column(
        DateTime(),
        default = pd.to_datetime(pa.now(), utc=True, dayfirst=True),
        comment= "Describes the time of the first data point in the data set. Use the ISO 8601:2004 date format",
        onupdate=pd.to_datetime(pa.now(), utc=True, dayfirst=True),
        nullable=True,
        )
    time_coverage_end = mapped_column(
        DateTime(),
        default = pd.to_datetime(pa.now(), utc=True, dayfirst=True),
        comment= "end of measurement date time in the data system",
        onupdate=pd.to_datetime(pa.now(), utc=True, dayfirst=True),
        nullable=True,
        )
    time_coverage_duration = mapped_column(
        String(),
        default = isodate.duration_isoformat(timedelta(minutes=10)),
        comment= "Describes the duration of the data set. Use ISO 8601:2004 duration format, preferably the extended format as recommended in the Attribute Content Guidance section. Examples: P1Y, P3M, P10D",
        nullable=True,
        )
    time_coverage_resolution = mapped_column(
        String(),
        default = isodate.duration_isoformat(timedelta(seconds=1)),
        comment= "Describes the targeted time period between each value in the data set. Use ISO 8601:2004 duration format, preferably the extended format as recommended in the Attribute Content Guidance section. Examples: P1Y, P3M, P10D",
        nullable=True,
        )
    project = mapped_column(
        Enum(projectEnum),
        default = "highre",
        comment = "The name of the project(s) principally responsible for originating this data",
        )
    program = mapped_column(
        String(),
        default = None,
        comment = "The overarching program(s) of which the dataset is a part",
        nullable=True,
        )
    title = mapped_column(
        String(),
        default = None,
        comment = "A short phrase or sentence describing the dataset",
        nullable=True,
        )
    processing_level = mapped_column(
        Enum(ProcesslevelsEnum),
        default = "L1b",
        comment = "A textual description of the  quality control level of the data",
        nullable=False,
        )
    uuid = mapped_column(
        String(),
        default = generate_uuid,
        comment = "Machine readable unique identifier for each netCDF file",
        nullable=False,
        )

    def __repr__(self):
        return "DataFile(date_created = '{self.date_created}', "\
            "date_issued = '{self.date_issued}', "\
            "date_modified = '{self.date_modified}', "\
            "date_metadata_modified = '{self.date_metadata_modified}', "\
            "time_coverage_start = '{self.time_coverage_start}', "\
            "time_coverage_end = '{self.time_coverage_end}', "\
            "time_coverage_duration = '{self.time_coverage_duration}', "\
            "time_coverage_resolution = '{self.time_coverage_resolution}', "\
            "project = '{self.project}', "\
            "program = '{self.program}', "\
            "title = '{self.title}', "\
            "processing_level = '{self.processing_level}', "\
            "uuid = '{self.uuid}')".format(self.self)


class Site(Base):
    __tablename__ = "site"
    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    name = mapped_column(String, comment="name of the site", nullable=False)
    region = mapped_column(
        String(),
        default="Bremerhaven, Germany",
        comment="region of Measurement",
        nullable=False,
    )
    site_code = mapped_column(
        Integer,
        default=None,
        comment="Reference Id for the site",
        nullable=True,
    )
    platform_code = mapped_column(
        String(),
        default=None,
        comment="Reference name and Id for the platform",
        nullable=True,
    )
    site_address = mapped_column(
        String(),
        default=
        "Am Luneort 15, Flughafen Bremerhaven, 27572 Bremerhaven, Germany",
        comment="Address of the test site / installation site",
        nullable=False,
    )
    geospatial_bounds = mapped_column(
        String(),
        default=None,
        comment="data's 2D or 3D geospatial extent",
        nullable=True,
    )
    geospatial_bounds_crs = mapped_column(
        String(),
        default="WGS 84 EPSG:4326",
        comment=
        "The coordinate reference system (CRS) of the point coordinates in the geospatial_bounds attribute (EPSG CRSs are strongly recommended)",
        nullable=True,
    )
    geospatial_bounds_vertical_crs = mapped_column(
        String(),
        default="EPSG:3855",
        comment=
        "The vertical coordinate reference system (CRS) for the Z axis of the point coordinates in the geospatial_bounds attribute",
        nullable=True,
    )
    geospatial_lat_min = mapped_column(
        String(),
        default=None,
        comment="minimum lattitude of the measurment site",
        nullable=True,
    )
    geospatial_lat_max = mapped_column(
        String(),
        default=None,
        comment="maximum lattitude of the measurment site",
        nullable=True,
    )
    geospatial_lon_min = mapped_column(
        String(),
        default=None,
        comment="minimum longitude of the measurement site",
        nullable=True,
    )
    geospatial_lon_max = mapped_column(
        String(),
        default=None,
        comment="maximum longitude of the measurement site",
        nullable=True,
    )
    geospatial_lat_units = mapped_column(
        String(),
        default="decimal_degrees_north",
        comment="units for latitude",
        nullable=False,
    )
    geospatial_lon_units = mapped_column(
        String(),
        default="decimal_degrees_east",
        comment="units for longitude",
        nullable=False,
    )
    geospatial_vertical_min = mapped_column(
        Integer(),
        default=None,
        comment="minumum vertical height of the measurement",
        nullable=True,
    )
    geospatial_vertical_max = mapped_column(
        Integer(),
        default=None,
        comment="maximum vertical height of the measurement",
        nullable=True,
    )
    geospatial_vertical_units = mapped_column(
        String(),
        default="meters",
        comment="units of vertical measurement",
        nullable=True,
    )
    geospatial_vertical_positive = mapped_column(
        String(),
        default="up",
        comment="direction of position vertical direction",
        nullable=True,
    )
    elevation = mapped_column(
        String(),
        default="0 m",
        comment="Elevation at the measurement site",
        nullable=True,
    )
    elevation_asl = mapped_column(
        String(),
        default="11 m",
        comment="Elevation of the measurement site above sea level",
        nullable=True,
    )
    surface_roughness = mapped_column(
        String(),
        default="0.1 m",
        comment="roughness length classification according to Davenport",
        nullable=True,
    )
    wind_power_class = mapped_column(
        Integer,
        default=3,
        comment="Wind power class according to wind potential",
        nullable=True,
    )
    terrain_slope = mapped_column(
        String(),
        default="1 degree",
        comment="Terrain slope at the measurement site",
        nullable=True,
    )
    area_type = mapped_column(
        String(),
        default="simple",
        comment="site classification",
        nullable=True,
    )
    mean_vegetation_height = mapped_column(
        String(),
        default="2 m",
        comment="mean height of the trees at the location",
        nullable=True,
    )
    mean_obstacles_height = mapped_column(
        String(),
        default="8 m",
        comment="mean height of the buildings at the location",
        nullable=True,
    )
    main_wind_direction_min = mapped_column(
        String(),
        default=None,
        comment="main wind direction min from North =0°",
        nullable=True,
    )
    main_wind_direction_max = mapped_column(
        String(),
        default=None,
        comment="main wind direction max from North =0°",
        nullable=True,
    )
    weibull_param_scale = mapped_column(
        Float,
        default=None,
        comment="scale parameter of weibull distribution from metmast (annual)",
        nullable=True,
    )
    weibull_param_shape = mapped_column(
        Float,
        default=None,
        comment="shape parameter of weibull distribution from metmast (annual)",
        nullable=True,
    )

    def __repr__(self):
        return "Site(region = '{self.region}', "\
            "site_code = '{self.site_code}', "\
            "platform_code = '{self.platform_code}', "\
            "site_address = '{self.site_address}', "\
            "geospatial_bounds = '{self.geospatial_bounds}', "\
            "geospatial_bounds_crs = '{self.geospatial_bounds_crs}', "\
            "geospatial_bounds_vertical_crs = '{self.geospatial_bounds_vertical_crs}', "\
            "geospatial_lat_min = '{self.geospatial_lat_min}', "\
            "geospatial_lat_max = '{self.geospatial_lat_max}', "\
            "geospatial_lon_min = '{self.geospatial_lon_min}', "\
            "geospatial_lon_max = '{self.geospatial_lon_max}', "\
            "geospatial_lat_units = '{self.geospatial_lat_units}', "\
            "geospatial_lon_units = '{self.geospatial_lon_units}', "\
            "geospatial_vertical_min = '{self.geospatial_vertical_min}', "\
            "geospatial_vertical_max = '{self.geospatial_vertical_max}', "\
            "geospatial_vertical_units = '{self.geospatial_vertical_units}', "\
            "geospatial_vertical_positive = '{self.geospatial_vertical_positive}', "\
            "elevation = '{self.elevation}', "\
            "elevation_asl = '{self.elevation_asl}', "\
            "surface_roughness = '{self.surface_roughness}', "\
            "wind_power_class = '{self.wind_power_class}', "\
            "terrain_slope = '{self.terrain_slope}', "\
            "area_type = '{self.area_type}', "\
            "mean_vegetation_height = '{self.mean_vegetation_height}', "\
            "mean_obstacles_height = '{self.mean_obstacles_height}', "\
            "main_wind_direction_min = '{self.main_wind_direction_min}', "\
            "main_wind_direction_max = '{self.main_wind_direction_max}', "\
            "weibull_param_scale = '{self.weibull_param_scale}', "\
            "weibull_param_shape = '{self.weibull_param_shape}')".format(self.self)


class WindFarm(Base):
    __tablename__ = "windfarm"
    wt_N = mapped_column(
        Integer(),
        default = 1,
        comment = "number of turbines",
        nullable=True,
    )
    wt_name = mapped_column(
        String(),
        default = "Adwen AD8-180",
        comment = "name of the wind turbine",
        nullable=True,
    )
    wt_make_version = mapped_column(
        String(),
        default = None,
        comment = "make version of the wind turbine",
        nullable=True,
    )
    wt_rated_power = mapped_column(
        String(),
        default = "3600 kW",
        comment = "rated power of the turbine",
        nullable=True,
    )
    wt_id = mapped_column(
        String(),
        default = None,
        unique=True,
        comment = "turbine id number",
    )
    wt_longitude = mapped_column(
        String(),
        default = None,
        comment = "longitude coordinate system",
        nullable=True,
    )
    wt_latitude = mapped_column(
        String(),
        default = None,
        comment = "latitude coordinate system",
        nullable=True,
    )
    wt_oem = mapped_column(
        String(),
        default = None,
        comment = "manufacturer of the wind turbine",
        nullable=True,
    )
    wt_install_date = mapped_column(
        String(),
        default = None,
        comment = "date of installation",
        nullable=True,
    )
    wt_rotor_diameter = mapped_column(
        String(),
        default = "180 m",
        comment = "rotor diameter",
        nullable=True,
    )
    wt_hub_height = mapped_column(
        String(),
        default = "115 m",
        comment = "hub height of the wind turbine",
        nullable=True,
    )
    wt_cutin = mapped_column(
        String(),
        default = "4 m/s",
        comment = "cutin wind speed above idling phase",
        nullable=True,
    )
    wt_cutout = mapped_column(
        String(),
        default = "25 m/s",
        nullable=True,
        comment = "cutout wind speed before transition phase",
    )
    wt_tilt_angle = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "tilt angle of the hub wrt horizontal",
    )
    wt_cone_angle = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "cone angle of the hub wrt horizontal",
    )
    wt_pitch_controller = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "type of pitch controller",
    )
    name = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = " name of the windfarm",
    )
    id = mapped_column(
        String(),
        default = None,
        primary_key = True,
        comment = "windfarm id number",
        nullable=True,
    )
    total_power = mapped_column(
        String(),
        default = None,
        comment = "Total power output of the wind farm",
        nullable=True,
    )
    longitude_min = mapped_column(
        String(),
        default = None,
        comment = "minimum longitude for the extent of the wind farm",
        nullable=True,
    )
    longitude_max = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "maximum longitude for the extent of the wind farm",
    )
    latitude_min = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "minimum latitude for the extent of the wind farm",
    )
    latitude_max = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "maximum latitude for the extent of the wind farm",
    )

    def __repr__(self):
        return "WindFarm(wt_N = '{self.wt_N}', "\
            "wt_name = '{self.wt_name}', "\
            "wt_make_version = '{self.wt_make_version}', "\
            "wt_rated_power = '{self.wt_rated_power}', "\
            "wt_id = '{self.wt_id}', "\
            "wt_longitude = '{self.wt_longitude}', "\
            "wt_latitude = '{self.wt_latitude}', "\
            "wt_oem = '{self.wt_oem}', "\
            "wt_install_date = '{self.wt_install_date}', "\
            "wt_rotor_diameter = '{self.wt_rotor_diameter}', "\
            "wt_hub_height = '{self.wt_hub_height}', "\
            "wt_cutin = '{self.wt_cutin}', "\
            "wt_cutout = '{self.wt_cutout}', "\
            "wt_tilt_angle = '{self.wt_tilt_angle}', "\
            "wt_cone_angle = '{self.wt_cone_angle}', "\
            "wt_pitch_controller = '{self.wt_pitch_controller}', "\
            "name = '{self.name}', "\
            "id = '{self.id}', "\
            "total_power = '{self.total_power}', "\
            "longitude_min = '{self.longitude_min}', "\
            "longitude_max = '{self.longitude_max}', "\
            "latitude_min = '{self.latitude_min}', "\
            "latitude_min = '{self.latitude_min}')".format(self.self)

class Documentation(Base):
    __tablename__ = "documentation"
    doc_id = mapped_column(Integer, primary_key=True, autoincrement=True)
    instruction_manual = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "Name of the instruction manual/report",
    )
    installation_manual = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "Name of the installation manual/report",
    )
    operation_manual = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "Name of the operation manual/report",
    )
    logbook = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "Name of the logbook filename",
    )
    support_manual = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "Name of the support manual/report",
    )
    network_manual = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "Name of the network access manual/report",
    )
    data_report = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "Name of the Data manual/report",
    )
    templates = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "Name of the Template documents",
    )
    repair_manual = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "Name of the repair report",
    )
    technical_description = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "Name of the technical specifications or description report",
    )
    calibration_report = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "Name of the calibration manual/report",
    )
    invoice_id = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "Invoice id of the sensor",
    )
    quantity = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "Number of sensors in the invoice",
    )
    service_name = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "any additional service included in the purchase agreement",
    )

    def __repr__(self):
        return "Documentation(instruction_manual = '{self.instruction_manual}', "\
            "installation_manual = '{self.installation_manual}', "\
            "operation_manual = '{self.operation_manual}', "\
            "logbook = '{self.logbook}', "\
            "support_manual = '{self.support_manual}', "\
            "network_manual = '{self.network_manual}', "\
            "date_report = '{self.date_report}', "\
            "templates = '{self.templates}', "\
            "repair_manual = '{self.repair_manual}', "\
            "technical_description = '{self.technical_description}', "\
            "calibration_report = '{self.calibration_report}', "\
            "invoice_id = '{self.invoice_id}', "\
            "quantity = '{self.quantity}', "\
            "service_name = '{self.service_name}', "\
            "license = '{self.license}')".format(self.self)

class InstrumentSpecs(Base):
    __tablename__ = "instrument_specs"
    manufacturer = mapped_column(
        Enum(InstrumentManufacturerEnum),
        default = "vaisala",
        comment = "Manufacturer of the sensor",
        nullable=False,
    )
    provider = mapped_column(
        Enum(InstrumentManufacturerEnum),
        default = "vaisala",
        nullable=True,
        comment = "provider of the sensor if deviating from the manufacturer, e.g. GWU providing Vaisala's WindCube",
    )
    sensor_id = mapped_column(
        String(),
        default = None,
        primary_key = True,
        comment = "Instrument id in system",
    )
    make_model = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "make of the instrument",
    )
    model = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "model version or name of the instrument",
    )
    size = mapped_column(
        ArrayType(),
        default = [1,1,1],
        nullable=True,
       comment = "Size in [Length, Breadth, Height] in metres",
    )
    weight = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "Weight in kg",
    )
    type = mapped_column(
        Enum(InstrumentEnum),
        default = None,
        nullable=False,
        comment = "global type of instrument",
    )
    serial_number = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "Serial number of the instrument",
    )
    operation = mapped_column(
        Enum(InstrumentSubcategoryEnum),
        default = None,
        comment = "Operation principles",
        nullable=True,
    )
    sensor_status_qc = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "Last known condition of the sensor",
    )
    power = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "Power supply details",
    )
    total_number = mapped_column(
        Integer(),
        default = None,
        nullable=True,
        comment = "total number of same sensor at the location",
    )
    category = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "Category of the sensor",
    )
    classification = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "Classification of the sensor",
    )
    # sensor_type = mapped_column(
    #     Enum(InstrumentEnum),
    #     default = None,
    #     comment = "Type of sensor for sensing"
    # )
    sensor_subtype = mapped_column(
        Enum(InstrumentSubcategoryEnum),
        default = None,
        nullable=True,
        comment = "subtype of the sensor type",
    )
    short_name = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "short name for the sensor",
    )
    long_name = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "long name for the sensor",
    )
    safety_check = mapped_column(
        String(),
        default = "safe",
        nullable=False,
        comment = "Safety during operation [safe / unsafe / restricted / not applicable]",
    )
    uuid = mapped_column(
        String(),
        default = generate_uuid(),
        comment = "unique id of the instruction manual",
    )
    ref_coord_sys = mapped_column(
        Enum(CoordinateSysEnum),
        default = 'lc',
        comment = "reference coordinate system for the global system",
        nullable=False,
    )
    measurement_station = mapped_column(
        Enum(MeasurementStationTypeEnum),
        default = "mast",
        nullable=False,
        comment = "Type of measurement station or the location of the installed sensor",
    )
    calibration_slope = mapped_column(
        Float(),
        default = None,
        nullable=True,
        comment = "standard calibration slope of sensors (e.g. wind tunnel calibration for cups)",
    )
    calibration_offset = mapped_column(
        Float(),
        default = None,
        nullable=True,
        comment = "standard calibration offset of sensors (e.g. wind tunnel calibration for cups)",
    )
    calibration_correlation = mapped_column(
        Float(),
        default = None,
        nullable=True,
        comment = "standard calibration correlation of sensors (e.g. wind tunnel calibration for cups)",
    )

    def __repr__(self):
        return "InstrumentSpecs(manufacturer = '{self.manufacturer}', "\
            "provider = '{self.provider}', "\
            "sensor_id = '{self.sensor_id}', "\
            "make_model = '{self.make_model}', "\
            "model = '{self.model}', "\
            "size = '{self.size}', "\
            "weight = '{self.weight}', "\
            "type = '{self.type}', "\
            "serial_number = '{self.serial_number}', "\
            "operation = '{self.operation}', "\
            "sensor_status_qc = '{self.sensor_status_qc}', "\
            "power = '{self.power}', "\
            "total_number = '{self.total_number}', "\
            "category = '{self.category}', "\
            "classification = '{self.classification}', "\
            "sensor_type = '{self.sensor_type}', "\
            "sensor_subtype = '{self.sensor_subtype}', "\
            "short_name = '{self.short_name}', "\
            "long_name = '{self.long_name}', "\
            "safety_check = '{self.safety_check}', "\
            "uuid = '{self.uuid}', "\
            "ref_coord_sys = '{self.ref_coord_sys}', "\
            "measurement_station = '{self.measurement_station}')".format(self.self)


class LidarConfig(Base):
    __tablename__ = "lidar_config"
    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    lidar_name = mapped_column(
        Enum(LidarNamesEnum),
        default = "srws2",
        nullable=False,
        comment = "name of the lidar, corresponding to the short_name in instrumentSpecs table",
    )
    wavelength = mapped_column(
        String(),
        default = "1565 nm",
        nullable=True,
        comment = "wavelength of the laser pulse in [nm]",
    )
    Pmax = mapped_column(
        String(),
        default = "1000 mW",
        nullable=True,
        comment = "maximum output power of the Laser",
    )
    prism_angle = mapped_column(
        String(),
        default = "30 deg",
        nullable=True,
        comment = "prism angle [degrees]",
    )
    focus_min = mapped_column(
        String(),
        default = "20 m",
        nullable=True,
        comment = "minimal focus distance [m]",
    )
    focus_max = mapped_column(
        String(),
        default = "300 m",
        nullable=True,
        comment = "maximum focus distance [m]",
    )
    laser_safety_class = mapped_column(
        Enum(LaserSafetyClassEnum),
        default = "4",
        nullable=False,
        comment = "laser safety class according to DIN EN 60825-12001-11",
    )
    lidar_type = mapped_column(
        Enum(LidarTypeEnum),
        default = "cw",
        nullable=False,
        comment = "type of wind lidar",
    )
    focal_length = mapped_column(
        String(),
        default = "580 mm",
        nullable=True,
        comment = "telescope focal length f0 [mm] between the prism and laser source",
    )
    aperture_radius_lens = mapped_column(
        String(),
        default = "56 mm",
        nullable=True,
        comment = "lens aperture radius a0 (I_a0/I0 = e^-2)",
    )
    beam_waist = mapped_column(
        String(),
        default = "0.88 mm",
        nullable=True,
        comment = "minimum beam waist at R =100m",
    )
    prisms = mapped_column(
        String(),
        default = "two co/counter axial prisms with third axis focus motor",
        nullable=True,
        comment = "prism information for lidar configuration",
    )
    spectra_N = mapped_column(
        Integer,
        default = 512,
        nullable=True,
        comment = "number of spectra for wind speed estimation",
    )
    pulse_duration = mapped_column(
        String(),
        default = "1e-7 s",
        nullable=True,
        comment = "pulse duration of Lidar [s]" ,
    )
    range_max = mapped_column(
        String(),
        default = "1000 m",
        nullable=True,
        comment = "maximum measurement range of the Lidar",
    )
    aperture_diameter_telescope = mapped_column(
        String(),
        default = "3-inch",
        nullable=True,
        comment = "aperture diameter of the lidar telescope",
    )

    def __repr__(self):
        return "LidarConfig(lidar_name = '{self.lidar_name}', " \
            "wavelength = '{self.wavelength}', " \
            "Pmax = '{self.Pmax}', " \
            "prism_angle = '{self.prism_angle}', " \
            "focus_min = '{self.focus_min}', " \
            "focus_max = '{self.focus_max}', " \
            "laser_safety_class = '{self.laser_safety_class}', " \
            "lidar_type = '{self.lidar_type}', " \
            "focal_length = '{self.focal_length}', " \
            "aperture_radius_lens = '{self.aperture_radius_lens}', " \
            "beam_waist = '{self.beam_waist}', " \
            "prisms = '{self.prisms}', " \
            "spectra_N = '{self.spectra_N}', " \
            "pulse_duration = '{self.pulse_duration}', " \
            "range_max = '{self.range_max}', " \
            "aperture_diameter_telescope = '{self.aperture_radius_telescope}')".format(self.self)

class AggQualityControl(Base):
    # aggregate quality control flag governing the whole dataset
    __tablename__ = "qc"
    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    flag_values = mapped_column(
        ArrayType(),
        default = "0b",
        nullable=True,
        comment = "Comma-separated list of map values. Flag_values maps each value in the variable to a value in the flag_meanings \
            in order to interpret the meaning of the value in the array"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ,
    )
    flag_masks = mapped_column(
        ArrayType(),
        default = "0b",
        nullable=True,
        comment = "Comma separated list of binary masks. The boolean conditions are identified by performing bitwise AND of the variable\
                value and the flag_masks. The data type of the mask must match the data type of the associated variable."                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               ,
    )
    flag_meanings = mapped_column(
        ArrayType(),
        default = "quality_good",
        nullable=True,
        comment = "Space-separated list of interpretations corresponding to each of the flag_values and/or flag_masks",
    )
    missing_value = mapped_column(
        Float(),
        default = np.nan,
        nullable=True,
        comment = "a scalar or vector containing values indicating conditions of missing data",
    )
    _FillValue = mapped_column(
        Float(),
        default = np.nan,
        nullable=True,
        comment = "NaN or -9999 or inf (outside the valid_range and actual_range",
    )
    def __repr__(self):
        return "AggQualityControl(flag_values = '{self.flag_values}', " \
            "flag_masks = '{self.flag_masks}', " \
            "flag_meanings = '{self.flag_meanings}', " \
            "missing_value = '{self.missing_value}', " \
            "_FillValue = '{self._FillValue}')".format(self.self)

class Variable(Base):
    __tablename__ = "variable"
    # https://git.earthdata.nasa.gov/projects/EMFD/repos/unified-metadata-model/browse/variable/v1.8.1/umm-var-json-schema.json#     __tablename__ = "variable"
    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    name = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "The name of a variable as given in data, by sensor manufacturer, or any other provider",
    )
    long_name = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "The long_name attribute contains a long descriptive name which may, for example, be used for labeling plots.",
    )
    standard_name = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "The name used to identify the physical quantity as per CF conventions. A standard name contains no whitespace and is case sensitive.",
    )
    # short_name = mapped_column()
    definition = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "The definition of the variable.",
    )
    description = mapped_column(
        String(),
        default = None,
        comment = "Long description of the variable. Typically in detail according to the application usage",
    )
    format = mapped_column(
        String(),
        default = None,
        comment="Describes the organization of the data content so that users and applications know how to read and use the content. The controlled vocabulary for formats is maintained in the Keyword Management System (KMS): https://gcmd.earthdata.nasa.gov/KeywordViewer/scheme/DataFormat?gtm_scheme=DataFormat",
    )
    units = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "The units associated with a variable",
    )
    data_type = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "Specify data type of a variable. These types can be either: uint8, uint16, etc.",
    )
    dimensions = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "A variable consists of one or more dimensions. An example of a dimension name is 'XDim'. An example of a dimension size is '1200'. Variables are rarely one dimensional.",
    )
    valid_range = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment= "Valid ranges of variable data values.",
    )
    scale_factor = mapped_column(
        Float(),
        default = 1,
        comment = "The scale is the numerical factor by which all values in the stored data field are multiplied in order to obtain the original values. May be used together with Offset. An example of a scale factor is '0.002'",
    )
    add_offset = mapped_column(
        Float(),
        default = 0,
        comment = "The offset is the value which is either added to or subtracted from all values in the stored data field in order to obtain the original values. May be used together with Scale. An example of an offset is '0.49'.",
    )
    _FillValue = mapped_column(
        Integer(),
        default = np.nan,
        comment = "The fill value of the variable in the data file. It is generally a value which falls outside the valid range. For example, if the valid range is '0, 360', the fill value may be '-1'. The fill value type is data provider-defined. For example, 'Out of Valid Range'.",
    )
    flag_values = mapped_column(
        String(),
        default = "0b",
        comment = "Comma-separated list of map values. Flag_values maps each value in the variable to a value in the flag_meanings \
            in order to interpret the meaning of the value in the array"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ,
    )
    flag_masks = mapped_column(
        String(),
        default = "0b",
        comment = "Comma separated list of binary masks. The boolean conditions are identified by performing bitwise AND of the variable\
                value and the flag_masks. The data type of the mask must match the data type of the associated variable."                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               ,
    )
    flag_meanings = mapped_column(
        String(),
        default = "quality_good",
        comment = "Space-separated list of interpretations corresponding to each of the flag_values and/or flag_masks",
    )
    missing_value = mapped_column(
        Float(),
        default = np.nan,
        comment = "a scalar or vector containing values indicating conditions of missing data",
    )

    def __repr__(self):
        return "Variable(name = 'self.name', ", \
            "long_name = '{self.name}', ", \
            "standard_name = '{self.standard_name}', ", \
            "definition = '{self.definition}', ", \
            "format = '{self.format}', ", \
            "units = '{self.units}', ", \
            "data_type = '{self.data_type}', ", \
            "dimensions = '{self.dimensions}', ", \
            "valid_range = '{self.valid_range}', ", \
            "scale_factor = {self.scale_factor}, ", \
            "add_offset = {self.add_offset}, ", \
            "_FillValue = {self._FillValue}, ", \
            "flag_values = '{self.flag_values}', ", \
            "flag_masks = '{self.flag_masks}', ", \
            "flag_meanings = '{self.flag_meanings}', ", \
            "missing_values = '{self.missing_values}')".format(self=self)

class OntolidarVariable(Base):
    # reference: https://github.com/IEA-Wind-Task-32/wind-lidar-ontology
    __tablename__ = "ontolidar_variable"
    uri = mapped_column(
        String(),
        default = None,
        primary_key = True,
        comment = "Unique resource identifier for a variable name. E.g.: ontolidar:VelocityAzimuthDisplay",
    )
    type = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "format type of the variable. E.g.: rdf, ttl, etc",
    )
    altLabel = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "an Abbreviation for including the variable in user work_flows. E.g.: vad",
        )
    broader = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "Parent group of the variable. E.g.: Windfield reconstruction",
        )
    narrower = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "children group of the variable. E.g.: pulsed-wave / continuous-wave",
        )
    definition = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "Definition of the variable. E.g.: VAD is a method of analyzing data from a complete conical scan whereby many closely spaced azimuthal points may be sampled by the lidar, and the data are used to estimate the wind speed at each height using a statistical fitting method.",
        )
    editorialNote = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "The VAD method is described in Lhermitte (1966) and Browning and Wexler (1968).",
    )
    inScheme = mapped_column(
        String(),
        default = 'http://vocab.ieawindtask32.org/wind-lidar-ontology/',
        comment = "Schematic of the ontology",
    )
    prefLabel = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "a long name of the variable E.g.Velocity azimuth display",
    )
    cf_standard_name = mapped_column(
        String(),
        default=None,
        comment = "The name used to identify the physical quantity as per CF conventions. A standard name contains no whitespace and is case sensitive.",
    )

    def __repr__(self):
        return "Ontolidar_Variable(uri = '{self.uri}', ", \
            "type = '{self.type}', ", \
            "altLabel = '{self.type}', ", \
            "broader = '{self.type}', ", \
            "narrower = '{self.type}', ", \
            "definition = '{self.type}', ", \
            "editorialNote = '{self.type}', ", \
            "inScheme = '{self.type}', ", \
            "prefLabel = '{self.type}', ", \
            "standard_name = '{self.type}')".format(self.self)

class CFvariable(Base):
    # standard variables from the cf standard names tables, extend to include the parent and children following the grammar rules
    #
    __tablename__ = "cf_variable"
    version_number = mapped_column(
        String(),
        default = '1.10',
        comment = "version of the CF conventions standard names table",
    )
    standard_name = mapped_column(
        String(),
        default = None,
        primary_key = True,
        comment = "standard name of the variable, stored as 'id' in standard tables.xml",
    )
    canonical_units = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "units of the variable",
    )
    symbol = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "amip values from standard names table representing the symbols used in meteorology",
    )
    description = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "description or definition of the variable",
    )
    alias = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "alias defined by CF standard conventions",
    )


class Logbook(Base):
    __tablename__ = "logbook"
    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    date = mapped_column(
        DateTime,
        default = pa.now(),
        comment = "Date time of the task performed for the sensor",
        onupdate=pa.now(),
    )
    task = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "details of the task performed",
    )
    sensor_name = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "Name of the sensor affected" ,
    )
    employee_name = mapped_column(
        Enum(peopleEnum),
        default = "ashim",
        comment = "name of the employee or contact person handling the task",
    )
    employee_url = mapped_column(
        String(),
        default=None,
        comment = "employee/company's webpage",
        )
    employee_email = mapped_column(
        String(),
        default = "ashim.giyanani@iwes.fraunhofer.de",
        comment = "email address of the employee",
        )
    employee_tel = mapped_column(
        String(),
        default = "+49 151 42462025",
        comment = "telephone contact of the employee",
        )
    employee_institution = mapped_column(
        String(),
        default = "Fraunhofer IWES",
        comment = "name of the company/institution",
        )
    technician_name = mapped_column(
        Enum(peopleEnum),
        default = "ashim",
        comment = "name of the employee or contact person handling the task",
    )
    technician_url = mapped_column(
        String(),
        default=None,
        comment = "employee/company's webpage",
        )
    technician_email = mapped_column(
        String(),
        default = "ashim.giyanani@iwes.fraunhofer.de",
        comment = "email address of the employee",
        )
    technician_tel = mapped_column(
        String(),
        default = "+49 151 42462025",
        comment = "telephone contact of the employee",
        )
    technician_institution = mapped_column(
        String(),
        default = "Fraunhofer IWES",
        comment = "name of the company/institution",
        )
    problem = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment ="describe the problem that generated the task to be performed",
    )
    details = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "description of the problem, task performed, solutions and future actions if any",
    )
    comments = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "general comments or remarks",
    )
    affects_data = mapped_column(
        Boolean,
        default = True,
        comment = "if the changes or the updates affect the data or not",
    )
    report_url = mapped_column(
        String(),
        default = None,
        nullable=True,
        comment = "path to the report generated or the link to the short note with details",
    )

    def __repr__(self):
        return "Logbook(date='{self.date}', ", \
                        "task = '{self.task}', " \
                        "sensor_name = '{self.sensor_name}', "\
                        "employee_name = '{self.employee_name}', "\
                        "employee_url = '{self.employee_url}', "\
                        "employee_email = '{self.employee_email}', "\
                        "employee_tel = '{self.employee_tel}', "\
                        "employee_institution = '{self.employee_institution}', "\
                        "technician_name = '{self.technician_name}', "\
                        "technician_url = '{self.technician_url}', "\
                        "technician_email = '{self.technician_email}', "\
                        "technician_tel = '{self.technician_tel}', "\
                        "technician_institution = '{self.technician_institution}', "\
                        "problem = '{self.problem}', "\
                        "details = '{self.details}', "\
                        "comments = '{self.comments}', "\
                        "affects_data = '{self.affects_data}', "\
                        "report_url = '{self.report_url}')".format(self=self)

class DublinCore(Base):
    __tablename__ = "dublincore"
    id = mapped_column(Integer, primary_key=True, autoincrement=True)
    abstract = mapped_column(
        String, 
        comment="A summary of the resource.", 
        default=None,
        )
    accessRights = mapped_column(
        String, 
        comment="Information about who access the resource or an indication of its security status. Access Rights may include information regarding access or restrictions based on privacy, security, or other policies.", 
        default=None,
        )
    accrualMethod = mapped_column(
        String, 
        comment="The method by which items are added to a collection. Recommended practice is to use a value from the Collection Description Accrual Method Vocabulary [DCMI-ACCRUALMETHOD].", 
        default=None,
        )
    accrualPeriodicity = mapped_column(
        String, 
        comment="The frequency with which items are added to a collection. Recommended practice is to use a value from the Collection Description Frequency Vocabulary [DCMI-COLLFREQ].", 
        default=None,
        )
    accrualPolicy = mapped_column(
        String, 
        comment="The policy governing the addition of items to a collection. Recommended practice is to use a value from the Collection Description Accrual Policy Vocabulary [DCMI-ACCRUALPOLICY].", 
        default=None,
        )
    alternative = mapped_column(
        String, 
        comment="An alternative name for the resource. The distinction between titles and alternative titles is application-specific.", 
        default=None,
        )
    audience = mapped_column(
        String, 
        comment="A class of agents for whom the resource is intended or useful.	Recommended practice is to use this property with non-literal values from a vocabulary of audience types.", 
        default=None,
        )
    available = mapped_column(
        String, 
        comment="Date that the resource became or will become available. Recommended practice is to describe the date, date/time, or period of time as recommended for the property Date, of which this is a subproperty.", 
        default=None,
        )
    bibliographicCitation = mapped_column(
        String, 
        comment="A bibliographic reference for the resource. Recommended practice is to include sufficient bibliographic detail to identify the resource as unambiguously as possible.", 
        default=None,
        )
    conformsTo = mapped_column(
        String, 
        comment="An established standard to which the described resource conforms.", 
        default=None,
        )
    contributor = mapped_column(
        String, 
        comment="An entity responsible for making contributions to the resource. The guidelines for using names of persons or organizations as creators apply to contributors.", 
        default=None,
        )
    coverage = mapped_column(
        String, 
        comment="The spatial or temporal topic of the resource, spatial applicability of the resource, or jurisdiction under which the resource is relevant.", 
        default=None,
        )
    created = mapped_column(
        String, 
        comment="Date of creation of the resource.", 
        default=None,
        )
    creator = mapped_column(
        String, 
        comment="An entity responsible for making the resource. Recommended practice is to identify the creator with a URI. ", 
        default=None,
        )
    date = mapped_column(
        String, 
        comment="A point or period of time associated with an event in the lifecycle of the resource. Recommended practice is to express the date, date/time, or period of time according to ISO 8601-1 [ISO 8601-1] ", 
        default=None,
        )
    dateAccepted = mapped_column(
        String, 
        comment="Date of copyright of the resource.", 
        default=None,
        )
    dateCopyrighted = mapped_column(
        String, 
        comment="Date of copyright of the resource.", 
        default=None,
        )
    dateSubmitted = mapped_column(
        String, 
        comment="Date of submission of the resource.", 
        default=None,
        )
    description = mapped_column(
        String,
        comment = "An account of the resource. Description may include but is not limited to: an abstract, a table of contents, a graphical representation, or a free-text account of the resource.",
        default=None
    )
    educationLevel = mapped_column(
        String, 
        comment="A class of agents, defined in terms of progression through an educational or training context, for which the described resource is intended.", 
        default=None,
        )
    extent = mapped_column(
        String, 
        comment="The size or duration of the resource. Recommended practice is to specify the file size in megabytes and duration in ISO 8601 format.", 
        default=None,
        )
    format = mapped_column(
        String, 
        comment="The file format, physical medium, or dimensions of the resource. Recommended practice is to use a controlled vocabulary where available. For example, for file formats one could use the list of Internet Media Types [MIME]. Examples of dimensions include size and duration.", 
        default=None,
        )
    hasFormat = mapped_column(
        String, 
        comment="A related resource that is substantially the same as the pre-existing described resource, but in another format.", 
        default=None,
        )
    hasPart = mapped_column(
        String, 
        comment="A related resource that is included either physically or logically in the described resource.", 
        default=None,
        )
    hasVersion = mapped_column(
        String, 
        comment="A related resource that is a version, edition, or adaptation of the described resource.", 
        default=None,
        )
    identifier = mapped_column(
        String, 
        comment="An unambiguous reference to the resource within a given context. Examples include International Standard Book Number (ISBN), Digital Object Identifier (DOI), and Uniform Resource Name (URN). Persistent identifiers should be provided as HTTP URIs.", 
        default=None,
        )
    instructionalMethod = mapped_column(
        String, 
        comment="A process, used to engender knowledge, attitudes and skills, that the described resource is designed to support.", 
        default=None,
        )
    isFormatOf = mapped_column(
        String, 
        comment="A pre-existing related resource that is substantially the same as the described resource, but in another format.", 
        default=None,
        )
    isPartOf = mapped_column(
        String, 
        comment="A related resource in which the described resource is physically or logically included.", 
        default=None,
        )
    isReferencedBy = mapped_column(
        String, 
        comment="A related resource that references, cites, or otherwise points to the described resource.", 
        default=None,
        )
    isReplacedBy = mapped_column(
        String, 
        comment="A related resource that supplants, displaces, or supersedes the described resource.", 
        default=None,
        )
    isRequiredBy = mapped_column(
        String, 
        comment="A related resource that requires the described resource to support its function, delivery, or coherence.", 
        default=None,
        )
    issued = mapped_column(
        String, 
        comment="Date of formal issuance of the resource.", 
        default=None,
        )
    isVersionOf = mapped_column(
        String, 
        comment="A related resource of which the described resource is a version, edition, or adaptation.", 
        default=None,
        )
    language = mapped_column(
        String, 
        comment="A language of the resource.", 
        default=None,
        )
    license = mapped_column(
        String, 
        comment="A legal document giving official permission to do something with the resource.", 
        default=None,
        )
    mediator = mapped_column(
        String, 
        comment="An entity that mediates access to the resource.", 
        default=None,
        )
    medium = mapped_column(
        String, 
        comment="The material or physical carrier of the resource.", 
        default=None,
        )
    modified = mapped_column(
        String, 
        comment="Date on which the resource was changed.", 
        default=None,
        )
    provenance = mapped_column(
        String, 
        comment="A statement of any changes in ownership and custody of the resource since its creation that are significant for its authenticity, integrity, and interpretation.", 
        default=None,
        )
    publisher = mapped_column(
        String, 
        comment="An entity responsible for making the resource available.", 
        default=None,
        )
    references = mapped_column(
        String, 
        comment="A related resource that is referenced, cited, or otherwise pointed to by the described resource.", 
        default=None,
        )
    relation = mapped_column(
        String, 
        comment="An entity responsible for making the resource available.", 
        default=None,
        )
    replaces = mapped_column(
        String, 
        comment="A related resource that is supplanted, displaced, or superseded by the described resource.", 
        default=None,
        )
    requires = mapped_column(
        String, 
        comment="A related resource that is required by the described resource to support its function, delivery, or coherence.", 
        default=None,
        )
    rights = mapped_column(
        String, 
        comment="Information about rights held in and over the resource.", 
        default=None,
        )
    rightsHolder = mapped_column(
        String, 
        comment="A person or organization owning or managing rights over the resource.", 
        default=None,
        )
    source = mapped_column(
        String, 
        comment="A related resource from which the described resource is derived.", 
        default=None,
        )
    spatial = mapped_column(
        String, 
        comment="Spatial characteristics of the resource.", 
        default=None,
        )
    subject = mapped_column(
        String, 
        comment="A topic of the resource.", 
        default=None,
        )
    tableOfContents = mapped_column(
        String, 
        comment="A list of subunits of the resource.", 
        default=None,
        )
    temporal = mapped_column(
        String,
        comment = "Temporal characteristics of the resource.",
        default=None
    )    

class DataAccessLayer:

    def __init__(self, memory=True):
        # self.connection = None
        self.engine = None
        self.session = None
        if not memory:
            self.conn_string = "sqlite+pysqlite:///metadata_gen.db"
        else:
            self.conn_string = "sqlite:///memory"

        # self.metadata = MetaData()

    def memory(self, Base=Base):
        self.engine = None
        self.session = None
        self.conn_string = "sqlite:///memory"
        self.engine = create_engine(self.conn_string, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def connect(self, Base=Base):
        # https://stackoverflow.com/a/34344200
        self.engine = create_engine(self.conn_string, echo=False)
        # Base.metadata.drop_all(self.engine)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

# dal = DataAccessLayer()
# dal.connect()

if __name__ == "__main__":
    from metadata_tables import *
    dal = DataAccessLayer()
    dal.connect()
    dal.session = dal.Session()
    u1 = User(name='ashim')

    dal.session.add_all([u1])
    dal.session.flush()
    dal.session.commit()
