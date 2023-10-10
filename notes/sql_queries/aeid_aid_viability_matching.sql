SELECT 
    aeid,
    assay_component_endpoint_name,
	assay.aid,
    assay_name,
    -- assay_component.acid,
    -- assay_component_name,
    assay_function_type
FROM
    prod_internal_invitrodb_v4_1.assay_component_endpoint
        JOIN
    assay_component ON assay_component_endpoint.acid = assay_component.acid
        JOIN
    assay ON assay_component.aid = assay.aid
WHERE
    aeid in (26, 38, 40, 46, 58 ,60)