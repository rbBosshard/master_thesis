SELECT 
    distinct mc5.aeid, dsstox_substance_id, hitc, modl, coff, sample.spid  -- use distinct on only dsstox_substance_id to filter out duplicates
FROM
    prod_internal_invitrodb_v4_1.mc5
        JOIN
    prod_internal_invitrodb_v4_1.mc4 ON mc5.m4id = mc4.m4id
        JOIN
    sample ON mc4.spid = sample.spid
        JOIN
    chemical ON sample.chid = chemical.chid
WHERE
    mc5.aeid = 54 AND hitc > 0.5;