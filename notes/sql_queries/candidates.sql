SELECT
    ace.aeid,
    ace.assay_component_endpoint_name,
    ace.analysis_direction,
    ace.signal_direction,
    a.assay_format_type
FROM
    invitrodb_v3o5.assay_component_endpoint AS ace
INNER JOIN
    assay_component AS ac ON ace.acid = ac.acid
INNER JOIN
    assay AS a ON ac.aid = a.AID
WHERE
    assay_function_type NOT IN ('background control') AND a.assay_format_type in ('cell-based');