# External Bulk Data in 3LC

This folder contains notebooks for showcasing different ways of externalizing bulk data in 3LC Tables.

__TLDR:__ you don't need to make redundant copies of your source data, just make
up a URL scheme and serve data over HTTP.

## Background

3LC Tables do not store "bulk data" in the Table parquet files themselves.
Instead, URLs pointing to their locations are included in the Table, and these
are requested by the Dashboard from the Object Service, which fetches the
content using UrlAdapters. The canonical example to have in mind is images:
paths to images in Tables, Object Service serving jpg/png bytes from disk/object
storage.

We recently extended the bulk data mechanism to support 2D and 3D pointclouds.
In the future we will generalize this further, but for now really only use this
mechanism for large lidar/radar pointcloud datasets. Only int32/float32 arrays
are supported, and the only way to create bulk data columns (without doing it
manually) is using the builtin schemas
`tlc.Geometry2/3DSchema(is_bulk_data=True)`.

Initially, this would only work when ingesting data, formatted as
`tlc.Geometry2D/3DInstances`-dataclasses, using a `tlc.TableWriter`. Internally,
data would be passed through a `BulkDataProcessor`, which enables the
redirection (writing arrays to .raw files and inserting Urls in the Table).

## Contents

The notebooks in this folder attempt to elucidate the mechanism by building up
understanding step by step.

[1](./1-externalize-manually-with-processor.ipynb) simply uses a
`BulkDataProcessor` directly, using an external write location. This is what
happens inside the `TableWriter` by default, except that data is always stored
relative to the Table being written (../../bulk_data).

[2](./2-externalize-manually-no-processor.ipynb) performs the logic done in the
BulkDataProcessor manually. This explains how the binary data property pairs
work ("x", "x_binary_property_url"), and also how data is packed in the raw
files (chunk indices and offset-lengths).

[3](./3-externalize-virtual-http-endpoints-single-attrs.ipynb) is where it gets
interesting. This example completely removes the need for writing any bulk data
to disk at all. Instead, at ingestion time, URLs are made up (based on some
identifying information in the source dataset). (If you don't need/want to
understand all the details, you can start here). When loading this Table in the
Dashboard, it is required to have a HTTP server running
[server](./http_server_single_attrs.py) to fetch data from the source dataset
and return in the wanted format.

[4](./4-externalize-virtual-http-endpoints-chunked.ipynb) is just a more
complicated version of 3, where multiple attributes from multiple samples are
batched together and returned in chunks. This requires the server to know which
mapping to use to map URLs to individual samples/attributes.

## Summary

We are hopeful that [3] will provide a useful pattern for customers who do not
want to make redundant copies of data and are comfortable hosting their own data
from their own data sources to be available for the 3LC Dashboard.

It will be natural to extend this example to allow cloud functions (lambdas) to
serve underlying data. This is the most portable and scalable solution. The work
for the customer lies in inventing a natural URL scheme for identifying data,
and then serving the required data in the required format. The 3LC Team will be
happy to assist with this.

> Note: the "redirect-external-data-to-http-server" pattern will work out of the
> box for image data also! This is relevant to users who can serve image-bytes
> on the fly, where the images might not be backed by file-paths in their
> natural state (e.g. images are stored in some company-wide black-box data
> service, or as frames in videos).

## Planned improvements

- We currently require bulk data urls to contain the `<start-byte>-<length>`
  suffix. In the future we will not require this to be present for
  single-attribute external data URLs (as in example [3]). In this case the user
  won't even need to load the source data at ingestion time.

- Up until now we have been working mostly with chunked raw files of size ~50MB.
  Now that we are moving towards single-attribute external data "files" (which
  is much more user friendly), we will need to do some optimizations to the
  pre-fetching in the 3LC Dashboard. This can be expected to be added shortly.
  Note that this could be made harder to do if the start-length suffix is
  omitted from the URLs, as BLOB-sizes will now be known upfront.
