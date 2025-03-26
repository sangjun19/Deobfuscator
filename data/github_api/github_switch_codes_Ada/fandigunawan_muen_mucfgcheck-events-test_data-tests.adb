// Repository: fandigunawan/muen
// File: tools/libmucfgcheck/tests/mucfgcheck-events-test_data-tests.adb

--  This package has been generated automatically by GNATtest.
--  You are allowed to add your code to the bodies of test routines.
--  Such changes will be kept during further regeneration of this file.
--  All code placed outside of test routine bodies will be lost. The
--  code intended to set up and tear down the test environment should be
--  placed into Mucfgcheck.Events.Test_Data.

with AUnit.Assertions; use AUnit.Assertions;
with System.Assertions;

--  begin read only
--  id:2.2/00/
--
--  This section can be used to add with clauses if necessary.
--
--  end read only
with Mucfgcheck.Validation_Errors;
--  begin read only
--  end read only
package body Mucfgcheck.Events.Test_Data.Tests is

--  begin read only
--  id:2.2/01/
--
--  This section can be used to add global variables and other elements.
--
--  end read only

--  begin read only
--  end read only

--  begin read only
   procedure Test_Physical_Event_Name_Uniqueness (Gnattest_T : in out Test);
   procedure Test_Physical_Event_Name_Uniqueness_5e2eca (Gnattest_T : in out Test) renames Test_Physical_Event_Name_Uniqueness;
--  id:2.2/5e2eca7d6fc14927/Physical_Event_Name_Uniqueness/1/0/
   procedure Test_Physical_Event_Name_Uniqueness (Gnattest_T : in out Test) is
--  end read only

      pragma Unreferenced (Gnattest_T);

      Data : Muxml.XML_Data_Type;
   begin
      Muxml.Parse (Data => Data,
                   Kind => Muxml.Format_B,
                   File => "data/test_policy.xml");

      --  Positive test, must not raise an exception.

      Physical_Event_Name_Uniqueness (XML_Data => Data);
      Assert (Condition => Validation_Errors.Is_Empty,
              Message   => "Unexpected error in positive test");

      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/events/event[@name='trap_to_sm']",
         Name  => "name",
         Value => "resume_linux");

      Physical_Event_Name_Uniqueness (XML_Data => Data);
      Assert (Condition => Validation_Errors.Contains
              (Msg => "Multiple physical events with name 'resume_linux'"),
              Message   => "Exception mismatch");
--  begin read only
   end Test_Physical_Event_Name_Uniqueness;
--  end read only


--  begin read only
   procedure Test_Source_Targets (Gnattest_T : in out Test);
   procedure Test_Source_Targets_dd485f (Gnattest_T : in out Test) renames Test_Source_Targets;
--  id:2.2/dd485fd3b78efbdb/Source_Targets/1/0/
   procedure Test_Source_Targets (Gnattest_T : in out Test) is
--  end read only

      pragma Unreferenced (Gnattest_T);

      Data : Muxml.XML_Data_Type;
   begin
      Muxml.Parse (Data => Data,
                   Kind => Muxml.Format_B,
                   File => "data/test_policy.xml");

      --  Positive test, must not raise an exception.

      Source_Targets (XML_Data => Data);
      Assert (Condition => Validation_Errors.Is_Empty,
              Message   => "Unexpected error in positive test");

      declare
         Node : DOM.Core.Node
           := DOM.Core.Documents.Create_Element
             (Doc      => Data.Doc,
              Tag_Name => "event");
         Target_Node : constant DOM.Core.Node
           := Muxml.Utils.Get_Element
             (Doc   => Data.Doc,
              XPath => "/system/subjects/subject[@name='linux']"
              & "/events/target");
      begin
         DOM.Core.Elements.Set_Attribute
           (Elem  => Node,
            Name  => "id",
            Value => "22");
         DOM.Core.Elements.Set_Attribute
           (Elem  => Node,
            Name  => "logical",
            Value => "system_reboot");
         DOM.Core.Elements.Set_Attribute
           (Elem  => Node,
            Name  => "physical",
            Value => "system_reboot");
         Node := DOM.Core.Nodes.Append_Child
           (N         => Target_Node,
            New_Child => Node);

         Source_Targets (XML_Data => Data);
         Assert (Condition => Validation_Errors.Contains
                 (Msg => "Invalid number of targets for kernel-mode event "
                  & "'system_reboot': 1 (no target allowed)"),
                 Message   => "Exception mismatch (target)");
            Node := DOM.Core.Nodes.Remove_Child
              (N         => Target_Node,
               Old_Child => Node);
      end;

      declare
         Node : constant DOM.Core.Node := Muxml.Utils.Get_Element
           (Doc   => Data.Doc,
            XPath => "/system/subjects/subject/events/target/"
            & "event[@physical='trap_to_sm']/..");
      begin
         Muxml.Utils.Remove_Child
           (Node       => Node,
            Child_Name => "event");

         Source_Targets (XML_Data => Data);
         Assert (Condition => Validation_Errors.Contains
                 (Msg => "Invalid number of targets for event 'trap_to_sm': 0"),
                 Message   => "Exception mismatch (target)");
      end;

      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/events/event[@name='trap_to_sm']",
         Name  => "name",
         Value => "new_event");

      Source_Targets (XML_Data => Data);
      Assert (Condition => Validation_Errors.Contains
              (Msg => "Invalid number of sources for event 'new_event': 0"),
              Message   => "Exception mismatch (source)");
--  begin read only
   end Test_Source_Targets;
--  end read only


--  begin read only
   procedure Test_Subject_Event_References (Gnattest_T : in out Test);
   procedure Test_Subject_Event_References_0768ea (Gnattest_T : in out Test) renames Test_Subject_Event_References;
--  id:2.2/0768eab62525b03d/Subject_Event_References/1/0/
   procedure Test_Subject_Event_References (Gnattest_T : in out Test) is
--  end read only

      pragma Unreferenced (Gnattest_T);

      Data : Muxml.XML_Data_Type;
   begin
      Muxml.Parse (Data => Data,
                   Kind => Muxml.Format_B,
                   File => "data/test_policy.xml");

      --  Positive test, must not raise an exception.

      Subject_Event_References (XML_Data => Data);
      Assert (Condition => Validation_Errors.Is_Empty,
              Message   => "Unexpected error in positive test");

      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/subjects/subject/events/target/"
         & "event[@physical='trap_to_sm']",
         Name  => "physical",
         Value => "nonexistent_dst");

      Subject_Event_References (XML_Data => Data);
      Assert (Condition => Validation_Errors.Contains
              (Msg => "Event 'nonexistent_dst' referenced by subject 'sm' does"
               & " not exist"),
              Message   => "Exception mismatch (target)");

      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/subjects/subject/events/source/group/"
         & "event[@physical='resume_linux']",
         Name  => "physical",
         Value => "nonexistent_src");

      Subject_Event_References (XML_Data => Data);
      Assert (Condition => Validation_Errors.Contains
              (Msg => "Event 'nonexistent_src' referenced by subject 'sm' does"
               & " not exist"),
              Message   => "Exception mismatch (source)");
--  begin read only
   end Test_Subject_Event_References;
--  end read only


--  begin read only
   procedure Test_Self_References (Gnattest_T : in out Test);
   procedure Test_Self_References_af5859 (Gnattest_T : in out Test) renames Test_Self_References;
--  id:2.2/af5859813505ea74/Self_References/1/0/
   procedure Test_Self_References (Gnattest_T : in out Test) is
--  end read only

      pragma Unreferenced (Gnattest_T);

      Data : Muxml.XML_Data_Type;
   begin
      Muxml.Parse (Data => Data,
                   Kind => Muxml.Format_B,
                   File => "data/test_policy.xml");

      --  Positive test, must not raise an exception.

      Self_References (XML_Data => Data);
      Assert (Condition => Validation_Errors.Is_Empty,
              Message   => "Unexpected error in positive test");

      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/subjects/subject/events/target/event"
         & "[@physical='linux_console']",
         Name  => "physical",
         Value => "linux_keyboard");

      Self_References (XML_Data => Data);
      Assert (Condition => Validation_Errors.Contains
              (Msg => "Reference to self in event 'linux_keyboard' of subject "
               & "'vt'"),
              Message   => "Exception mismatch");

      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/subjects/subject/events/target/event"
         & "[@physical='linux_keyboard']",
         Name  => "physical",
         Value => "linux_console");

      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/events/event[@name='linux_console']",
         Name  => "mode",
         Value => "self");

      Self_References (XML_Data => Data);
      Assert (Condition => Validation_Errors.Contains
              (Msg => "Reference to other subject in self-event "
               & "'linux_console' of subject 'linux'"),
              Message   => "Exception mismatch");
--  begin read only
   end Test_Self_References;
--  end read only


--  begin read only
   procedure Test_Switch_Same_Core (Gnattest_T : in out Test);
   procedure Test_Switch_Same_Core_9bc636 (Gnattest_T : in out Test) renames Test_Switch_Same_Core;
--  id:2.2/9bc636b0bd4cd54e/Switch_Same_Core/1/0/
   procedure Test_Switch_Same_Core (Gnattest_T : in out Test) is
--  end read only

      pragma Unreferenced (Gnattest_T);

      Data : Muxml.XML_Data_Type;
   begin
      Muxml.Parse (Data => Data,
                   Kind => Muxml.Format_B,
                   File => "data/test_policy.xml");

      --  Positive test, must not raise an exception.

      Switch_Same_Core (XML_Data => Data);
      Assert (Condition => Validation_Errors.Is_Empty,
              Message   => "Unexpected error in positive test");


      --  Switch between subjects on different CPUs.

      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/subjects/subject/events/source/group/"
         & "event[@physical='linux_keyboard']",
         Name  => "physical",
         Value => "resume_linux");

      Switch_Same_Core (XML_Data => Data);
      Assert (Condition => Validation_Errors.Contains
              (Msg => "Destination subject 'linux' (CPU 1) in subject's 'vt' "
                    & "(CPU 0) switch notification 'resume_linux' invalid - "
                    & "must run on the same CPU and be in the same scheduling "
                    & "group"),
              Message   => "Exception mismatch (1)");

      --  Error must still be raised even if multiple sources exist and the
      --  first one is valid while the second is not.

      declare

         SM_Node : constant DOM.Core.Node := Muxml.Utils.Get_Element
           (Doc   => Data.Doc,
            XPath => "/system/subjects/subject[@name='sm']");
         Subjects : constant DOM.Core.Node := Muxml.Utils.Get_Element
           (Doc   => Data.Doc,
            XPath => "/system/subjects");
         VT_Node : constant DOM.Core.Node
           := DOM.Core.Nodes.Remove_Child
             (N         => Subjects,
              Old_Child => Muxml.Utils.Get_Element
                (Doc   => Data.Doc,
                 XPath => "/system/subjects/subject[@name='vt']"));
      begin
         Muxml.Utils.Append_Child (Node      => Subjects,
                                   New_Child => VT_Node);

         Switch_Same_Core (XML_Data => Data);
         Assert (Condition => Validation_Errors.Contains
                 (Msg => "Destination subject 'linux' (CPU 1) in subject's 'vt' "
                  & "(CPU 0) switch notification 'resume_linux' invalid - "
                  & "must run on the same CPU and be in the same scheduling "
                  & "group"),
                 Message   => "Exception mismatch (2)");
      end;

      --  Switch between subjects in different scheduling groups.

      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/subjects/subject[@name='vt']",
         Name  => "cpu",
         Value => "1");

      Switch_Same_Core (XML_Data => Data);
      Assert (Condition => Validation_Errors.Contains
              (Msg => "Destination subject 'linux' (CPU 1) in subject's 'vt' "
               & "(CPU 1) switch notification 'resume_linux' invalid - "
               & "must run on the same CPU and be in the same scheduling "
               & "group"),
              Message   => "Exception mismatch (3)");
--  begin read only
   end Test_Switch_Same_Core;
--  end read only


--  begin read only
   procedure Test_IPI_Different_Core (Gnattest_T : in out Test);
   procedure Test_IPI_Different_Core_c8a75b (Gnattest_T : in out Test) renames Test_IPI_Different_Core;
--  id:2.2/c8a75bb306ee763d/IPI_Different_Core/1/0/
   procedure Test_IPI_Different_Core (Gnattest_T : in out Test) is
--  end read only

      pragma Unreferenced (Gnattest_T);

      Data : Muxml.XML_Data_Type;
   begin
      Muxml.Parse (Data => Data,
                   Kind => Muxml.Format_B,
                   File => "data/test_policy.xml");

      --  Positive test, must not raise an exception.

      IPI_Different_Core (XML_Data => Data);
      Assert (Condition => Validation_Errors.Is_Empty,
              Message   => "Unexpected error in positive test");

      declare
         Evt_Node : constant DOM.Core.Node := Muxml.Utils.Get_Element
           (Doc   => Data.Doc,
            XPath => "/system/subjects/subject[@name='sm']"
            & "/events/source/group[@name='vmcall']");
         Node : DOM.Core.Node
           := DOM.Core.Documents.Create_Element
             (Doc      => Data.Doc,
              Tag_Name => "event");
      begin
         DOM.Core.Elements.Set_Attribute
           (Elem  => Node,
            Name  => "id",
            Value => "22");
         DOM.Core.Elements.Set_Attribute
           (Elem  => Node,
            Name  => "logical",
            Value => "foo");
         DOM.Core.Elements.Set_Attribute
           (Elem  => Node,
            Name  => "physical",
            Value => "linux_keyboard");
         Node := DOM.Core.Nodes.Append_Child
           (N         => Evt_Node,
            New_Child => Node);

         IPI_Different_Core (XML_Data => Data);
         Assert (Condition => Validation_Errors.Contains
                 (Msg => "Destination subject 'linux' (CPU 1) in subject's 'sm'"
                  & " (CPU 1) ipi notification 'linux_keyboard' invalid - must"
                  & " run on different CPU"),
                 Message   => "Exception mismatch (masked ipi target error)");
      end;

      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/events/event[@name='trap_to_sm']",
         Name  => "mode",
         Value => "ipi");

      IPI_Different_Core (XML_Data => Data);

      Assert (Condition => Validation_Errors.Contains
              (Msg => "Destination subject 'sm' (CPU 1) in subject's 'linux' "
               & "(CPU 1) ipi notification 'trap_to_sm' invalid - must run"
               & " on different CPU"),
              Message   => "Exception mismatch");
--  begin read only
   end Test_IPI_Different_Core;
--  end read only


--  begin read only
   procedure Test_Target_Event_ID_Name_Uniqueness (Gnattest_T : in out Test);
   procedure Test_Target_Event_ID_Name_Uniqueness_4511b9 (Gnattest_T : in out Test) renames Test_Target_Event_ID_Name_Uniqueness;
--  id:2.2/4511b9efaf3f0c9b/Target_Event_ID_Name_Uniqueness/1/0/
   procedure Test_Target_Event_ID_Name_Uniqueness (Gnattest_T : in out Test) is
--  end read only

      pragma Unreferenced (Gnattest_T);
      Data : Muxml.XML_Data_Type;
   begin
      Muxml.Parse (Data => Data,
                   Kind => Muxml.Format_B,
                   File => "data/test_policy.xml");

      --  Positive test, must not raise an exception.

      Target_Event_ID_Name_Uniqueness (XML_Data => Data);
      Assert (Condition => Validation_Errors.Is_Empty,
              Message   => "Unexpected error in positive test");

      --  Set duplicate event name.

      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/subjects/subject/events/target/event"
         & "[@physical='resume_linux']",
         Name  => "logical",
         Value => "channel_event_linux_keyboard");

      Target_Event_ID_Name_Uniqueness (XML_Data => Data);
      Assert (Condition => Validation_Errors.Contains
              (Msg => "Subject 'linux' has multiple target events with the same"
               & " name: 'channel_event_linux_keyboard'"),
              Message   => "Exception mismatch (1)");
      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/subjects/subject/events/target/event"
         & "[@physical='resume_linux']",
         Name  => "logical",
         Value => "resume_after_trap");

      --  Set duplicate event ID.

      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/subjects/subject/events/target/event"
         & "[@logical='resume_after_trap']",
         Name  => "id",
         Value => "1");
      Target_Event_ID_Name_Uniqueness (XML_Data => Data);
      Assert (Condition => Validation_Errors.Contains
              (Msg => "Subject 'linux' target events 'resume_after_trap' and "
               & "'channel_event_linux_keyboard' share ID 1"),
              Message   => "Exception mismatch (2)");
--  begin read only
   end Test_Target_Event_ID_Name_Uniqueness;
--  end read only


--  begin read only
   procedure Test_Source_Group_Event_ID_Name_Uniqueness (Gnattest_T : in out Test);
   procedure Test_Source_Group_Event_ID_Name_Uniqueness_27cefa (Gnattest_T : in out Test) renames Test_Source_Group_Event_ID_Name_Uniqueness;
--  id:2.2/27cefab96498a26d/Source_Group_Event_ID_Name_Uniqueness/1/0/
   procedure Test_Source_Group_Event_ID_Name_Uniqueness (Gnattest_T : in out Test) is
--  end read only

      pragma Unreferenced (Gnattest_T);

      Data : Muxml.XML_Data_Type;
   begin
      Muxml.Parse (Data => Data,
                   Kind => Muxml.Format_B,
                   File => "data/test_policy.xml");

      --  Positive test, must not raise an exception.

      Source_Group_Event_ID_Name_Uniqueness (XML_Data => Data);
      Assert (Condition => Validation_Errors.Is_Empty,
              Message   => "Unexpected error in positive test (1)");

      --  Set duplicate event name in different event group, must not raise an
      --  exception.

      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/subjects/subject/events/source/group/event"
         & "[@logical='default_event_0']",
         Name  => "logical",
         Value => "unmask_irq_60");
      Source_Group_Event_ID_Name_Uniqueness (XML_Data => Data);
      Assert (Condition => Validation_Errors.Is_Empty,
              Message   => "Unexpected error in positive test (2)");

      --  Set duplicate event name in one event group.

      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/subjects/subject/events/source/group/event"
         & "[@logical='unmask_irq_59']",
         Name  => "logical",
         Value => "unmask_irq_60");
      Source_Group_Event_ID_Name_Uniqueness (XML_Data => Data);
      Assert (Condition => Validation_Errors.Contains
              (Msg => "Subject 'linux' has multiple source events with the same"
               &" name: 'unmask_irq_60'"),
              Message   => "Exception mismatch (1)");

      --  Set duplicate event ID.

      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/subjects/subject/events/source/group/event"
         & "[@logical='resume_linux']",
         Name  => "id",
         Value => "1");

      Source_Group_Event_ID_Name_Uniqueness (XML_Data => Data);
      Assert (Condition => Validation_Errors.Contains
              (Msg => "Subject 'sm' source events 'resume_linux' and "
               & "'channel_event_sm_console' share ID 1"),
              Message   => "Exception mismatch (2)");
--  begin read only
   end Test_Source_Group_Event_ID_Name_Uniqueness;
--  end read only


--  begin read only
   procedure Test_Source_Group_Event_ID_Validity (Gnattest_T : in out Test);
   procedure Test_Source_Group_Event_ID_Validity_ed9d9b (Gnattest_T : in out Test) renames Test_Source_Group_Event_ID_Validity;
--  id:2.2/ed9d9bbe36269c5a/Source_Group_Event_ID_Validity/1/0/
   procedure Test_Source_Group_Event_ID_Validity (Gnattest_T : in out Test) is
--  end read only

      pragma Unreferenced (Gnattest_T);

      Data : Muxml.XML_Data_Type;
   begin
      Muxml.Parse (Data => Data,
                   Kind => Muxml.Format_B,
                   File => "data/test_policy.xml");

      --  Positive test, must not raise an exception.

      Source_Group_Event_ID_Validity (XML_Data => Data);
      Assert (Condition => Validation_Errors.Is_Empty,
              Message   => "Unexpected error in positive test");

      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/subjects/subject/events/source/group/event"
         & "[@logical='resume_linux']",
         Name  => "id",
         Value => "256");

      Source_Group_Event_ID_Validity (XML_Data => Data);
      Assert (Condition => Validation_Errors.Contains
              (Msg => "Subject 'sm': ID 256 of event 'resume_linux' invalid "
               & "for group VMCALL"),
              Message   => "Exception mismatch");
--  begin read only
   end Test_Source_Group_Event_ID_Validity;
--  end read only


--  begin read only
   procedure Test_Source_VMX_Exit_Event_Completeness (Gnattest_T : in out Test);
   procedure Test_Source_VMX_Exit_Event_Completeness_98714a (Gnattest_T : in out Test) renames Test_Source_VMX_Exit_Event_Completeness;
--  id:2.2/98714a72f6bc4eef/Source_VMX_Exit_Event_Completeness/1/0/
   procedure Test_Source_VMX_Exit_Event_Completeness (Gnattest_T : in out Test) is
--  end read only

      pragma Unreferenced (Gnattest_T);

      Data : Muxml.XML_Data_Type;
   begin
      Muxml.Parse (Data => Data,
                   Kind => Muxml.Format_B,
                   File => "data/test_policy.xml");

      --  Positive test, must not raise an exception.

      Source_VMX_Exit_Event_Completeness (XML_Data => Data);
      Assert (Condition => Validation_Errors.Is_Empty,
              Message   => "Unexpected error in positive test");

      Muxml.Utils.Remove_Elements
        (Doc   => Data.Doc,
         XPath => "/system/subjects/subject[@name='sm']/events/source/"
         & "group[@name='vmx_exit']/event[@id='23']");

      Source_VMX_Exit_Event_Completeness (XML_Data => Data);
      Assert (Condition => Validation_Errors.Contains
              (Msg => "Subject 'sm' does not specify 'vmx_exit' group source "
               & "event with ID 23"),
              Message   => "Exception mismatch (1)");

      Muxml.Utils.Add_Child
        (Parent     => Muxml.Utils.Get_Element
           (Doc   => Data.Doc,
            XPath => "/system/subjects/subject[@name='sm']/events/source/"
            & "group[@name='vmx_exit']"),
         Child_Name => "default");

      --  Must not raise an exception because of <default/> presence.

      Validation_Errors.Clear;
      Source_VMX_Exit_Event_Completeness (XML_Data => Data);
      Assert (Condition => Validation_Errors.Is_Empty,
              Message   => "Unexpected error in positive test");

      Muxml.Utils.Remove_Elements
        (Doc   => Data.Doc,
         XPath => "/system/subjects/subject[@name='sm']/events/source/"
         & "group[@name='vmx_exit']");

      Source_VMX_Exit_Event_Completeness (XML_Data => Data);
      Assert (Condition => Validation_Errors.Contains
              (Msg => "Subject 'sm' does not specify any source event in "
               & "'vmx_exit' group"),
              Message   => "Exception mismatch (2)");
--  begin read only
   end Test_Source_VMX_Exit_Event_Completeness;
--  end read only


--  begin read only
   procedure Test_Self_Event_Action (Gnattest_T : in out Test);
   procedure Test_Self_Event_Action_e649a6 (Gnattest_T : in out Test) renames Test_Self_Event_Action;
--  id:2.2/e649a6f8cf4efeb5/Self_Event_Action/1/0/
   procedure Test_Self_Event_Action (Gnattest_T : in out Test) is
--  end read only

      pragma Unreferenced (Gnattest_T);

      Data : Muxml.XML_Data_Type;
   begin
      Muxml.Parse (Data => Data,
                   Kind => Muxml.Format_B,
                   File => "data/test_policy.xml");

      --  Positive test, must not raise an exception.

      Self_Event_Action (XML_Data => Data);
      Assert (Condition => Validation_Errors.Is_Empty,
              Message   => "Unexpected error in positive test");

      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/subjects/subject/events/target/event"
         & "[@physical='linux_console']",
         Name  => "physical",
         Value => "linux_keyboard");
      Muxml.Utils.Remove_Child
        (Node       => Muxml.Utils.Get_Element
           (Doc   => Data.Doc,
            XPath => "/system/subjects/subject/events/target/event"
            & "[@physical='linux_keyboard']"),
         Child_Name => "inject_interrupt");
      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/events/event[@name='linux_keyboard']",
         Name  => "mode",
         Value => "self");

      Self_Event_Action (XML_Data => Data);
      Assert (Condition => Validation_Errors.Contains
              (Msg => "Self-event 'channel_event_linux_console' of subject "
               & "'vt' does not specify an action"),
              Message   => "Exception mismatch");
--  begin read only
   end Test_Self_Event_Action;
--  end read only


--  begin read only
   procedure Test_Kernel_Mode_Event_Actions (Gnattest_T : in out Test);
   procedure Test_Kernel_Mode_Event_Actions_f55e89 (Gnattest_T : in out Test) renames Test_Kernel_Mode_Event_Actions;
--  id:2.2/f55e893967a529e0/Kernel_Mode_Event_Actions/1/0/
   procedure Test_Kernel_Mode_Event_Actions (Gnattest_T : in out Test) is
--  end read only

      pragma Unreferenced (Gnattest_T);

      Data : Muxml.XML_Data_Type;

      ----------------------------------------------------------------------

      procedure Missing_System_Poweroff
      is
         Data : Muxml.XML_Data_Type;
      begin
         Muxml.Parse (Data => Data,
                      Kind => Muxml.Format_B,
                      File => "data/test_policy.xml");

         Muxml.Utils.Remove_Child
           (Node       => Muxml.Utils.Get_Element
              (Doc   => Data.Doc,
               XPath => "/system/subjects/subject/events/source/group/"
               & "event[system_poweroff]"),
            Child_Name => "system_poweroff");

         Kernel_Mode_Event_Actions (XML_Data => Data);
         Assert (Condition => Validation_Errors.Contains
                 (Msg => "Kernel-mode source event 'system_poweroff' of subject"
                  & " 'vt' does not specify mandatory event action"),
                 Message   => "Exception mismatch (Poweroff)");
      end Missing_System_Poweroff;

      ----------------------------------------------------------------------

      procedure Missing_System_Panic
      is
         Data : Muxml.XML_Data_Type;
      begin
         Muxml.Parse (Data => Data,
                      Kind => Muxml.Format_B,
                      File => "data/test_policy.xml");

         Muxml.Utils.Remove_Child
           (Node       => Muxml.Utils.Get_Element
              (Doc   => Data.Doc,
               XPath => "/system/subjects/subject[@name='vt']/events/source/"
               & "group/event[@logical='panic_0']"),
            Child_Name => "system_panic");

         Kernel_Mode_Event_Actions (XML_Data => Data);
         Assert (Condition => Validation_Errors.Contains
                 (Msg => "Kernel-mode source event 'panic_0' of subject"
                  & " 'vt' does not specify mandatory event action"),
                 Message   => "Exception mismatch (Panic)");
      end Missing_System_Panic;

      ----------------------------------------------------------------------

      procedure Missing_System_Reboot
      is
         Data : Muxml.XML_Data_Type;
      begin
         Muxml.Parse (Data => Data,
                      Kind => Muxml.Format_B,
                      File => "data/test_policy.xml");

         Muxml.Utils.Remove_Child
           (Node       => Muxml.Utils.Get_Element
              (Doc   => Data.Doc,
               XPath => "/system/subjects/subject/events/source/group/"
               & "event[system_reboot]"),
            Child_Name => "system_reboot");

         Kernel_Mode_Event_Actions (XML_Data => Data);
         Assert (Condition => Validation_Errors.Contains
                 (Msg => "Kernel-mode source event 'system_reboot' of subject "
                  & "'vt' does not specify mandatory event action"),
                 Message   => "Exception mismatch (Reboot)");
      end Missing_System_Reboot;

      ----------------------------------------------------------------------

      procedure Missing_Unmask_Irq
      is
         Data : Muxml.XML_Data_Type;
      begin
         Muxml.Parse (Data => Data,
                      Kind => Muxml.Format_B,
                      File => "data/test_policy.xml");

         Muxml.Utils.Remove_Child
           (Node       => Muxml.Utils.Get_Element
              (Doc   => Data.Doc,
               XPath => "/system/subjects/subject/events/source/group/"
               & "event[unmask_irq]"),
            Child_Name => "unmask_irq");

         Kernel_Mode_Event_Actions (XML_Data => Data);
         Assert (Condition => Validation_Errors.Contains
                 (Msg => "Kernel-mode source event 'unmask_irq_57' of "
                  & "subject 'vt' does not specify mandatory event action"),
                 Message   => "Exception mismatch (Unmask IRQ)");
      end Missing_Unmask_Irq;

      ----------------------------------------------------------------------

      procedure Positive_Test
      is
         Data : Muxml.XML_Data_Type;
      begin
         Muxml.Parse (Data => Data,
                      Kind => Muxml.Format_B,
                      File => "data/test_policy.xml");

         --  Positive test, must not raise an exception.

         Kernel_Mode_Event_Actions (XML_Data => Data);
         Assert (Condition => Validation_Errors.Is_Empty,
                 Message   => "Unexpected error in positive test");
      end Positive_Test;
   begin
      Positive_Test;
      Missing_System_Panic;
      Missing_System_Poweroff;
      Missing_System_Reboot;
      Missing_Unmask_Irq;
--  begin read only
   end Test_Kernel_Mode_Event_Actions;
--  end read only


--  begin read only
   procedure Test_Kernel_Mode_System_Actions (Gnattest_T : in out Test);
   procedure Test_Kernel_Mode_System_Actions_150fed (Gnattest_T : in out Test) renames Test_Kernel_Mode_System_Actions;
--  id:2.2/150fed25899be466/Kernel_Mode_System_Actions/1/0/
   procedure Test_Kernel_Mode_System_Actions (Gnattest_T : in out Test) is
--  end read only

      pragma Unreferenced (Gnattest_T);

      Data : Muxml.XML_Data_Type;
   begin
      Muxml.Parse (Data => Data,
                   Kind => Muxml.Format_B,
                   File => "data/test_policy.xml");

      --  Positive test, must not raise an exception.

      Kernel_Mode_System_Actions (XML_Data => Data);
      Assert (Condition => Validation_Errors.Is_Empty,
              Message   => "Unexpected error in positive test");

      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/events/event[@name='system_reboot']",
         Name  => "mode",
         Value => "ipi");

      Kernel_Mode_System_Actions (XML_Data => Data);
      Assert (Condition => Validation_Errors.Contains
              (Msg => "System action for event 'system_reboot' of subject 'vt'"
               & " does not reference physical kernel-mode event "
               & "'system_reboot'"),
              Message   => "Exception mismatch (1)");

      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/events/event[@name='system_reboot']",
         Name  => "mode",
         Value => "kernel");
      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/events/event[@name='system_poweroff']",
         Name  => "mode",
         Value => "ipi");

      Kernel_Mode_System_Actions (XML_Data => Data);
      Assert (Condition => Validation_Errors.Contains
              (Msg => "System action for event 'system_poweroff' of subject "
               & "'vt' does not reference physical kernel-mode event "
               & "'system_poweroff'"),
              Message   => "Exception mismatch (2)");

      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/events/event[@name='system_poweroff']",
         Name  => "mode",
         Value => "kernel");
      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/events/event[@name='system_panic']",
         Name  => "mode",
         Value => "ipi");

      Kernel_Mode_System_Actions (XML_Data => Data);
      Assert (Condition => Validation_Errors.Contains
              (Msg => "System action for event 'panic_0' of subject 'tau0' does"
               & " not reference physical kernel-mode event 'system_panic'"),
              Message   => "Exception mismatch (3)");

      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/events/event[@name='system_panic']",
         Name  => "mode",
         Value => "kernel");
      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/events/event[@name='subject_sleep']",
         Name  => "mode",
         Value => "ipi");

      Kernel_Mode_System_Actions (XML_Data => Data);
      Assert (Condition => Validation_Errors.Contains
              (Msg => "System action for event 'HLT' of subject 'vt' does "
               & "not reference physical kernel-mode event "
               & "'subject_sleep'"),
              Message   => "Exception mismatch (4)");

      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/events/event[@name='subject_sleep']",
         Name  => "mode",
         Value => "kernel");
      Muxml.Utils.Set_Attribute
        (Doc   => Data.Doc,
         XPath => "/system/events/event[@name='subject_yield']",
         Name  => "mode",
         Value => "ipi");
      Kernel_Mode_System_Actions (XML_Data => Data);
      Assert (Condition => Validation_Errors.Contains
              (Msg => "System action for event 'PAUSE' of subject 'vt' does "
               & "not reference physical kernel-mode event "
               & "'subject_yield'"),
              Message   => "Exception mismatch (5)");
--  begin read only
   end Test_Kernel_Mode_System_Actions;
--  end read only


--  begin read only
   procedure Test_Level_Triggered_Unmask_IRQ_Action (Gnattest_T : in out Test);
   procedure Test_Level_Triggered_Unmask_IRQ_Action_03dce8 (Gnattest_T : in out Test) renames Test_Level_Triggered_Unmask_IRQ_Action;
--  id:2.2/03dce8c1382eca13/Level_Triggered_Unmask_IRQ_Action/1/0/
   procedure Test_Level_Triggered_Unmask_IRQ_Action (Gnattest_T : in out Test) is
--  end read only

      pragma Unreferenced (Gnattest_T);


      ----------------------------------------------------------------------

      procedure Missing_Event
      is
         Data : Muxml.XML_Data_Type;
      begin
         Muxml.Parse (Data => Data,
                      Kind => Muxml.Format_B,
                      File => "data/test_policy.xml");

         Muxml.Utils.Remove_Child
           (Node       => Muxml.Utils.Get_Element
              (Doc   => Data.Doc,
               XPath => "/system/subjects/subject/events/source/group/"
               & "event[unmask_irq]"),
            Child_Name => "unmask_irq");

         Level_Triggered_Unmask_IRQ_Action (XML_Data => Data);
         Assert (Condition => Validation_Errors.Contains
                 (Msg => "No event with unmask_irq action and matching number "
                  & "21 for IRQ 'wireless->irq'"),
                 Message   => "Exception mismatch (Missing event)");
      end Missing_Event;

      ----------------------------------------------------------------------

      procedure Positive_Test
      is
         Data : Muxml.XML_Data_Type;
      begin
         Muxml.Parse (Data => Data,
                      Kind => Muxml.Format_B,
                      File => "data/test_policy.xml");

         --  Positive test, must not raise an exception.

         Level_Triggered_Unmask_IRQ_Action (XML_Data => Data);
         Assert (Condition => Validation_Errors.Is_Empty,
                 Message   => "Unexpected error in positive test");
      end Positive_Test;

      ----------------------------------------------------------------------

      procedure Unassigned_IRQ_Event
      is
         Data : Muxml.XML_Data_Type;
      begin
         Muxml.Parse (Data => Data,
                      Kind => Muxml.Format_B,
                      File => "data/test_policy.xml");

         Muxml.Utils.Remove_Elements
           (Doc   => Data.Doc,
            XPath => "/system/subjects/subject/devices/"
            & "device[@physical='wireless']/irq");

         Level_Triggered_Unmask_IRQ_Action (XML_Data => Data);
         Assert (Condition => Validation_Errors.Contains
                 (Msg => "Event unmask_irq_57 of subject 'vt' has unmask_irq "
                  & "action for unassigned IRQ 21"),
                 Message   => "Exception mismatch (Unassigned IRQ)");
      end Unassigned_IRQ_Event;

      ----------------------------------------------------------------------

      procedure Unmask_Nr_Mismatch
      is
         Data : Muxml.XML_Data_Type;
      begin
         Muxml.Parse (Data => Data,
                      Kind => Muxml.Format_B,
                      File => "data/test_policy.xml");

         Muxml.Utils.Set_Attribute
           (Doc   => Data.Doc,
            XPath => "/system/subjects/subject/events/source/group/"
            & "event[@logical='unmask_irq_57']/unmask_irq",
            Name  => "number",
            Value => "0");

         Level_Triggered_Unmask_IRQ_Action (XML_Data => Data);
         Assert (Condition => Validation_Errors.Contains
                 (Msg => "No event with unmask_irq action and matching number "
                  & "21 for IRQ 'wireless->irq'"),
                 Message   => "Exception mismatch (Number mismatch)");
      end Unmask_Nr_Mismatch;
   begin
      Positive_Test;
      Missing_Event;
      Unmask_Nr_Mismatch;
      Unassigned_IRQ_Event;
--  begin read only
   end Test_Level_Triggered_Unmask_IRQ_Action;
--  end read only

--  begin read only
--  id:2.2/02/
--
--  This section can be used to add elaboration code for the global state.
--
begin
--  end read only
   null;
--  begin read only
--  end read only
end Mucfgcheck.Events.Test_Data.Tests;
