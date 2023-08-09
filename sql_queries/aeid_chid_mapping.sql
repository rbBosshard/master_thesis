-- set DBMS conenction read timeout interval (in seconds): 1000
Select * from
(
Select distinct mc4.aeid, chid from (
SELECT aeid, spid FROM invitrodb_v3o5.mc4) as mc4
Join sample on mc4.spid = sample.spid) as aeid_chid_map
