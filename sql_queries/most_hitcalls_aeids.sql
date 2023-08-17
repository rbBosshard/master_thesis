SELECT aeid, COUNT(*) as hitc_count
FROM invitrodb_v3o5.mc5
WHERE hitc = 1
GROUP BY aeid
ORDER BY hitc_count DESC;