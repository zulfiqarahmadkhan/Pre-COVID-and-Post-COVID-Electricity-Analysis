3
(�]`d  �               @   sH   d Z ddlmZ ddlmZ ddlmZ ddlZG dd� dejd�ZdS )	z)Base Class of Experiment Data Access API.�    )�absolute_import)�division)�print_functionNc               @   s    e Zd ZdZejddd��ZdS )�BaseExperimentz&Base class for experiment data access.NFc             C   s   dS )uP  Export scalar data as a pandas.DataFrame.

        Args:
          runs_filter: A regex filter for runs (e.g., r'run_[2-4]'). Operates in
            logical AND relation with `tags_filter`.
          tags_filter: A regex filter for tags (e.g., r'.*loss.*'). Operates in
            logical AND related with `runs_filter`.
          pivot: Whether to returned DataFrame will be pivoted (via pandas’
            `pivot_data()` method to a “wide” format wherein the tags of a
            given run and a given step are all collected in a single row.
            Setting `pivot` to `True` stipulates that the sets of step values
            are identical 