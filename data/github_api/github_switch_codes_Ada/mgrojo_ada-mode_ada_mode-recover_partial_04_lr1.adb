// Repository: mgrojo/ada-mode
// File: test/ada_mode-recover_partial_04_lr1.adb

-- From a real editing session, with partial parse. Encountered "error during resume"

--EMACSCMD:(setq wisi-indent-region-fallback nil)
--EMACSCMD:(switch-to-lr1)
end Process_Node;
begin
   if Cur.Node /= null then
      Process_Node (Cur);
   end if;
