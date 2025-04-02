
#include <xcdf/utility/XCDFUtility.h>
#include <xcdf/utility/EventSelectExpression.h>
#include <xcdf/utility/HistogramFiller.h>
#include <xcdf/utility/Histogram.h>
#include <xcdf/XCDFDefs.h>
#include <xcdf/version.h>

#include <set>
#include <sstream>

void Info(std::vector<std::string>& infiles) {

  XCDFFile f;
  if (infiles.size() == 0) {
    //read from stdin
    f.Open(std::cin);
  } else {
    f.Open(infiles[0], "r");
  }

  unsigned maxNameWidth = 0;
  unsigned maxParentWidth = 0;

  for (std::vector<XCDFFieldDescriptor>::const_iterator
                         it = f.FieldDescriptorsBegin();
                         it != f.FieldDescriptorsEnd(); ++it) {

    if ((it->name_).size() > maxNameWidth) {
      maxNameWidth = (it->name_).size();
    }

    if ((it->parentName_).size() > maxParentWidth) {
      maxParentWidth = (it->parentName_).size();
    }
  }

  if (maxNameWidth < 8) {
    maxNameWidth = 8;
  }

  maxParentWidth++;
  if (maxParentWidth < 8) {
    maxParentWidth = 8;
  }

  std::cout << std::endl;

  std::cout << std::setw(maxNameWidth) << "Field" << " " << std::setw(17) <<
             "Type" << " " << std::setw(11) << "Resolution" <<
                 std::setw(maxParentWidth) << "Parent" << " " <<
                 std::setw(10) << "Bytes" << " " << std::setw(10) <<
                 "Min" << " " << std::setw(10) << "Max" << std::endl;

  std::cout << std::setw(maxNameWidth) << "-----" << " " << std::setw(17) <<
               "----" << " " << std::setw(11) << "----------" <<
                   std::setw(maxParentWidth) << "------" << " " <<
                   std::setw(10) << "----" << " " << std::setw(10) <<
                   "---" << " " << std::setw(10) << "---" << std::endl;

  for (std::vector<XCDFFieldDescriptor>::const_iterator
                         it = f.FieldDescriptorsBegin();
                         it != f.FieldDescriptorsEnd(); ++it) {

    std::cout << std::setw(maxNameWidth) << it->name_ << " ";

    switch (it->type_) {

      case XCDF_UNSIGNED_INTEGER:
        std::cout << std::setw(17) << "Unsigned Integer" << 
                            " " << std::setw(11) << it->rawResolution_;
        break;
      case XCDF_SIGNED_INTEGER:
        std::cout << std::setw(17) << "Signed Integer" << " " <<
                     std::setw(11) <<
                       XCDFSafeTypePun<uint64_t, int64_t>(it->rawResolution_);
        break;
      case XCDF_FLOATING_POINT:
        std::cout << std::setw(17) << "Floating Point" << " " << 
                     std::setw(11) <<
                       XCDFSafeTypePun<uint64_t, double>(it->rawResolution_);
        break;

    }

    std::cout << std::setw(maxParentWidth) << it->parentName_ << " " <<
        std::setw(10) << f.GetFieldBytes(it->name_) << " " << std::setw(10);

    switch (it->type_) {

      case XCDF_UNSIGNED_INTEGER:
        std::cout << f.GetUnsignedIntegerFieldRange(it->name_).first << " " <<
            std::setw(10) << f.GetUnsignedIntegerFieldRange(it->name_).second;
        break;
      case XCDF_SIGNED_INTEGER:
        std::cout << f.GetSignedIntegerFieldRange(it->name_).first << " " <<
            std::setw(10) << f.GetSignedIntegerFieldRange(it->name_).second;
        break;
      case XCDF_FLOATING_POINT:
        std::cout << f.GetFloatingPointFieldRange(it->name_).first << " " <<
            std::setw(10) << f.GetFloatingPointFieldRange(it->name_).second;
        break;

    }

    std::cout << std::endl;
  }

  // Get the list of hard aliases in the header
  std::set<XCDFAliasDescriptor> headerDescriptors;
  headerDescriptors.insert(f.AliasDescriptorsBegin(), f.AliasDescriptorsEnd());

  // Load the event count to force load any aliases in trailers
  uint64_t eventCount = f.GetEventCount();

  unsigned maxAliasNameWidth = 6;
  for (std::vector<XCDFAliasDescriptor>::const_iterator
                          it = f.AliasDescriptorsBegin();
                          it != f.AliasDescriptorsEnd(); ++it) {

     if ((it->GetName()).size() > maxAliasNameWidth) {
       maxAliasNameWidth = (it->GetName()).size();
     }
  }

  if (f.AliasDescriptorsBegin() != f.AliasDescriptorsEnd()) {
    std::cout << "\n" << std::setw(maxAliasNameWidth) << "Alias" << " " <<
                  std::setw(17) << "Type" << " " << "Expression" << std::endl;

    std::cout << std::setw(maxAliasNameWidth) << "-----" << " " <<
                    std::setw(17) << "----" << " " << "----------" << std::endl;

    for (std::vector<XCDFAliasDescriptor>::const_iterator
                            it = f.AliasDescriptorsBegin();
                            it != f.AliasDescriptorsEnd(); ++it) {

      std::cout << std::setw(maxAliasNameWidth) << it->GetName() << " ";

      switch (it->GetType()) {

        case XCDF_UNSIGNED_INTEGER:
          std::cout << std::setw(17) << "Unsigned Integer" << " ";
          break;
        case XCDF_SIGNED_INTEGER:
          std::cout << std::setw(17) << "Signed Integer" << " ";
          break;
        case XCDF_FLOATING_POINT:
          std::cout << std::setw(17) << "Floating Point" << " ";
          break;
      }

      std::string expOut = "'" + it->GetExpression() + "'";
      if (headerDescriptors.find(*it) == headerDescriptors.end()) {
        expOut += "  (soft)";
      }

      std::cout << expOut << std::endl;
    }
  }

  std::cout << std::endl << "Entries: " << eventCount << std::endl;

  std::cout << std::endl << "Comments:" <<
               std::endl << "---------" << std::endl;

  f.LoadComments();
  for (std::vector<std::string>::const_iterator
                                   it = f.CommentsBegin();
                                   it != f.CommentsEnd(); ++it) {
    std::cout << *it << std::endl;
  }
}

void Dump(std::vector<std::string>& infiles) {

  uint64_t count = 0;
  XCDFFile f;
  for (unsigned i = 0; i <= infiles.size(); ++i) {

    if (i == infiles.size()) {
      if (infiles.size() == 0) {
        //read from stdin
        f.Open(std::cin);
      } else {
        continue;
      }
    } else {
      f.Open(infiles[i], "r");
    }

    while (f.Read()) {

      std::cout << "Event: " << count << std::endl;
      count++;
      std::cout << "------ " << std::endl;

      // Print out data from each field
      DumpFieldVisitor dumpFieldVisitor;
      f.ApplyFieldVisitor(dumpFieldVisitor);
      std::cout << std::endl;
    }

    std::cout << std::endl << "Comments:" <<
                 std::endl << "---------" << std::endl;

    f.LoadComments();
    for (std::vector<std::string>::const_iterator
                                  it = f.CommentsBegin();
                                  it != f.CommentsEnd(); ++it) {
      std::cout << *it << std::endl;
    }

    f.Close();
  }
}

void CSV(std::vector<std::string>& infiles) {

  XCDFFile f;
  for (unsigned i = 0; i <= infiles.size(); ++i) {

    if (i == infiles.size()) {
      if (infiles.size() == 0) {
        //read from stdin
        f.Open(std::cin);
      } else {
        continue;
      }
    } else {
      f.Open(infiles[i], "r");
    }

    if (i == 0) {
      PrintFieldNameVisitor printFieldNameVisitor(f);
      f.ApplyFieldVisitor(printFieldNameVisitor);
      std::cout << std::endl;
    }

    PrintFieldDataVisitor printFieldDataVisitor;
    while (f.Read()) {

      printFieldDataVisitor.Reset();
      f.ApplyFieldVisitor(printFieldDataVisitor);
      std::cout << std::endl;
    }

    f.Close();
  }
}

void Count(std::vector<std::string>& infiles,
           std::string& exp) {

  uint64_t count = 0;
  XCDFFile f;
  for (unsigned i = 0; i <= infiles.size(); ++i) {

    if (i == infiles.size()) {
      if (infiles.size() == 0) {
        //read from stdin
        f.Open(std::cin);
      } else {
        continue;
      }
    } else {
      f.Open(infiles[i], "r");
    }

    if (!exp.compare("")) {

      // just count the events
      count += f.GetEventCount();
    } else {
      // use the supplied expression
      EventSelectExpression expression(exp, f);
      while (f.Read()) {
        if (expression.SelectEvent()) {
          ++count;
        }
      }
    }
    f.Close();
  }

  std::cout << count << std::endl;
}

void Check(std::vector<std::string>& infiles) {

  XCDFFile f;
  for (unsigned i = 0; i <= infiles.size(); ++i) {

    if (i == infiles.size()) {
      if (infiles.size() == 0) {
        //read from stdin
        f.Open(std::cin);
      } else {
        continue;
      }
    } else {
      f.Open(infiles[i], "r");
    }

    // Allow internal checksum verification to detect errors
    while (f.Read()) { /* Do nothing */ }
    f.Close();
  }
}

std::set<std::string> ParseCSV(std::string& exp) {

  std::set<std::string> fields;

  char* expPtr = const_cast<char*>(exp.c_str());
  for (char* tok = strtok(expPtr, ","); tok != NULL;
                                             tok = strtok(NULL, ",")) {

    std::string str(tok);

    // Trim leading and trailing whitespace
    size_t endPos = str.find_last_not_of(" \n\r\t");
    if(endPos != std::string::npos) {
      str = str.substr(0, endPos+1);
    }

    size_t startPos = str.find_first_not_of(" \n\r\t");
    if(startPos != std::string::npos) {
      str = str.substr(startPos);
    }

    fields.insert(str);
  }
  return fields;
}

void CopyComments(XCDFFile& destination,
                  XCDFFile& source) {

  source.LoadComments();
  for (std::vector<std::string>::const_iterator
                             it = source.CommentsBegin();
                             it != source.CommentsEnd(); ++it) {

    destination.AddComment(*it);
  }
}

void CopyAliases(XCDFFile& destination,
                 XCDFFile& source,
                 std::string exclude = "") {

  for (std::vector<XCDFAliasDescriptor>::const_iterator
                             it = source.AliasDescriptorsBegin();
                             it != source.AliasDescriptorsEnd(); ++it) {

    try {
      // We might add duplicate aliases, so catch that here
      if (!destination.HasAlias(it->GetName()) && it->GetName() != exclude) {
        destination.CreateAlias(it->GetName(), it->GetExpression());
      }
    } catch (XCDFException& e) { }
  }
}

void SelectFields(std::vector<std::string>& infiles,
                  std::ostream& out,
                  std::string& exp,
                  std::string& concatArgs) {

  XCDFFile outFile(out);
  outFile.AddComment(concatArgs);
  std::set<std::string> fieldSpecs = ParseCSV(exp);
  std::set<std::string> fields;

  FieldCopyBuffer buf(outFile);

  // Spin through the files and copy the data
  XCDFFile f;
  for (unsigned i = 0; i <= infiles.size(); ++i) {

    if (i == infiles.size()) {
      if (infiles.size() == 0) {
        //read from stdin
        f.Open(std::cin);
      } else {
        continue;
      }
    } else {
      f.Open(infiles[i], "r");
    }

    // Build the list of fields
    if (fields.size() == 0) {
      MatchFieldsVisitor match(fieldSpecs);
      f.ApplyFieldVisitor(match);
      fields = match.GetMatches();
      if (fields.size() == 0) {
        XCDFFatal("Unable to match any fields from expression \"" <<
                                                         exp << "\"");
      }
    }

    // Check that the file contains all the fields
    for (std::set<std::string>::iterator it = fields.begin();
                                         it != fields.end(); ++it) {
      if (!f.HasField(*it)) {
        XCDFFatal("Unable to select field \"" <<
                               *it << "\": Field not present");
      }
    }

    SelectFieldVisitor selectFieldVisitor(f, fields, buf);
    f.ApplyFieldVisitor(selectFieldVisitor);

    while (f.Read()) {

      // Copy the data
      buf.CopyData();
      outFile.Write();
    }

    CopyComments(outFile, f);
    f.Close();
  }

  outFile.Close();
}

void Select(std::vector<std::string>& infiles,
            std::ostream& out,
            std::string& exp,
            std::string& concatArgs) {

  XCDFFile outFile(out);
  outFile.AddComment(concatArgs);

  FieldCopyBuffer buf(outFile);

  // Spin through the files and copy the data
  XCDFFile f;
  for (unsigned i = 0; i <= infiles.size(); ++i) {

    if (i == infiles.size()) {
      if (infiles.size() == 0) {
        //read from stdin
        f.Open(std::cin);
      } else {
        continue;
      }
    } else {
      f.Open(infiles[i], "r");
    }

    // Get the names of all the fields
    std::set<std::string> fields;
    GetFieldNamesVisitor getFieldNamesVisitor(fields);
    f.ApplyFieldVisitor(getFieldNamesVisitor);

    // Load the fields into the buffer for copying
    SelectFieldVisitor selectFieldVisitor(f, fields, buf);
    f.ApplyFieldVisitor(selectFieldVisitor);

    EventSelectExpression expression(exp, f);

    // Need to copy at beginning to ensure all known aliases are
    // placed into the header of the new file if at all possible
    CopyAliases(outFile, f);
    while (f.Read()) {

      // Check the expression; copy the data if true
      if (expression.SelectEvent()) {
        buf.CopyData();
        outFile.Write();
      }
    }

    CopyComments(outFile, f);
    // Copy any aliases unavailable at beginning
    CopyAliases(outFile, f);
    f.Close();
  }

  outFile.Close();
}

void Compare(const std::string& fileName1,
             const std::string& fileName2) {

  try {

    XCDFFile file1(fileName1.c_str(), "r");
    XCDFFile file2(fileName2.c_str(), "r");

    // load file comparison classes
    FileCompare compare1, compare2;
    file1.ApplyFieldVisitor(compare1);
    file2.ApplyFieldVisitor(compare2);

    // check field types, resolutions
    if (compare1.CompareFields(compare2)) {
      std::cout << "Files have fields with differing type or resolution\n";
      return;
    }

    // check that event counts are the same
    if (file1.GetEventCount() != file2.GetEventCount()) {
      std::cout << "Files have differing numbers of events\n";
      return;
    }

    // check event data
    uint64_t max = file1.GetEventCount();
    for (uint64_t i = 0; i < max; ++i) {
      file1.Read();
      file2.Read();
      if (compare1.CompareData(compare2)) {
        std::cout << "Event: " << i <<
                       ": Files have fields with differing data\n";
        return;
      }
    }
  } catch (XCDFException& e) {
    std::cout << "An error ocurred reading one of the files. Quitting"
                                                           << std::endl;
  }
}

void Recover(std::vector<std::string>& infiles,
             std::ostream& out) {

  XCDFFile f;
  if (infiles.size() == 0) {
    f.Open(std::cin);
  } else if (infiles.size() == 1) {
    // Open the file in recovery mode.  This avoids seeking to the
    // block table when reading a normal XCDF file, which immediately
    // ends the recovery.
    f.Open(infiles[0], "c");
  } else {
    std::cerr << "Only one input file is allowed for recover. Quitting"
                                                            << std::endl;
    exit(1);
  }

  XCDFFile outFile(out);

  try {

    // Get the names of all the fields
    std::set<std::string> fields;
    GetFieldNamesVisitor getFieldNamesVisitor(fields);
    f.ApplyFieldVisitor(getFieldNamesVisitor);

    // Load the fields into the buffer for copying
    FieldCopyBuffer buf(outFile);
    SelectFieldVisitor selectFieldVisitor(f, fields, buf);
    f.ApplyFieldVisitor(selectFieldVisitor);

    CopyAliases(outFile, f);
    while (f.Read()) {

      // Copy the data if true
      buf.CopyData();
      outFile.Write();
    }
  } catch (XCDFException& e) {
    std::cerr << "Corrupt file: Recovered " << outFile.GetEventCount()
                                               << " events." << std::endl;
  }

  try {
    CopyComments(outFile, f);
    CopyAliases(outFile, f);
    f.Close();
  } catch (XCDFException& e) { }
  outFile.Close();
}

void RemoveComments(std::vector<std::string>& infiles,
                    std::ostream& out) {

  XCDFFile f;
  if (infiles.size() == 0) {
    f.Open(std::cin);
  } else if (infiles.size() == 1) {
    f.Open(infiles[0], "r");
  } else {
    std::cerr << "Only one input file is allowed for remove-comments."
                                              " Quitting" << std::endl;
    exit(1);
  }

  XCDFFile outFile(out);

  // Get the names of all the fields
  std::set<std::string> fields;
  GetFieldNamesVisitor getFieldNamesVisitor(fields);
  f.ApplyFieldVisitor(getFieldNamesVisitor);

  // Load the fields into the buffer for copying
  FieldCopyBuffer buf(outFile);
  SelectFieldVisitor selectFieldVisitor(f, fields, buf);
  f.ApplyFieldVisitor(selectFieldVisitor);

  CopyAliases(outFile, f);
  while (f.Read()) {

    // Copy the data if true
    buf.CopyData();
    outFile.Write();
  }
  CopyAliases(outFile, f);
  outFile.Close();
}

bool IsStdout(std::ostream& out) {
  return &out == &std::cout;
}

void RemoveAlias(std::vector<std::string>& infiles,
              std::ostream& out,
              const std::string& name) {

  // If no output file is supplied, and input files are supplied,
  // attempt an in-place update
  if (IsStdout(out) && infiles.size() > 0) {
    for (unsigned i = 0; i < infiles.size(); ++i) {
      ModifyTrailer(infiles[i], AliasRemover(name), 3);
    }
    return;
  }

  // Rewrite the data to the specified output, with the alias removed
  XCDFFile outFile(out);
  FieldCopyBuffer buf(outFile);

  // Spin through the files and copy the data
  XCDFFile f;
  for (unsigned i = 0; i <= infiles.size(); ++i) {

    if (i == infiles.size()) {
      if (infiles.size() == 0) {
        //read from stdin
        f.Open(std::cin);
      } else {
        continue;
      }
    } else {
      f.Open(infiles[i], "r");
    }

    // Get the names of all the fields
    std::set<std::string> fields;
    GetFieldNamesVisitor getFieldNamesVisitor(fields);
    f.ApplyFieldVisitor(getFieldNamesVisitor);

    // Load the fields into the buffer for copying
    SelectFieldVisitor selectFieldVisitor(f, fields, buf);
    f.ApplyFieldVisitor(selectFieldVisitor);

    // Need to copy at beginning to ensure all known aliases are
    // placed into the header of the new file if at all possible
    CopyAliases(outFile, f, name);
    while (f.Read()) {
      buf.CopyData();
      outFile.Write();
    }

    CopyComments(outFile, f);
    // Copy any aliases unavailable at beginning
    CopyAliases(outFile, f, name);
    f.Close();
  }

  outFile.Close();
}

void AddAlias(std::vector<std::string>& infiles,
              std::ostream& out,
              const std::string& name,
              const std::string& expression) {

  // If no output file is supplied, and input files are supplied,
  // attempt an in-place update
  if (IsStdout(out) && infiles.size() > 0) {
    for (unsigned i = 0; i < infiles.size(); ++i) {
      ModifyTrailer(infiles[i], AliasAdder(name, expression), 3);
    }
    return;
  }

  // Rewrite the data to the specified output, with the new alias
  XCDFFile outFile(out);
  outFile.CreateAlias(name, expression);
  FieldCopyBuffer buf(outFile);

  // Spin through the files and copy the data
  XCDFFile f;
  for (unsigned i = 0; i <= infiles.size(); ++i) {

    if (i == infiles.size()) {
      if (infiles.size() == 0) {
        //read from stdin
        f.Open(std::cin);
      } else {
        continue;
      }
    } else {
      f.Open(infiles[i], "r");
    }

    // Get the names of all the fields
    std::set<std::string> fields;
    GetFieldNamesVisitor getFieldNamesVisitor(fields);
    f.ApplyFieldVisitor(getFieldNamesVisitor);

    // Load the fields into the buffer for copying
    SelectFieldVisitor selectFieldVisitor(f, fields, buf);
    f.ApplyFieldVisitor(selectFieldVisitor);

    // Need to copy at beginning to ensure all known aliases are
    // placed into the header of the new file if at all possible
    CopyAliases(outFile, f);
    while (f.Read()) {
      buf.CopyData();
      outFile.Write();
    }

    CopyComments(outFile, f);
    // Copy any aliases unavailable at beginning
    CopyAliases(outFile, f);
    f.Close();
  }

  outFile.Close();
}

void AddComment(std::vector<std::string>& infiles,
                std::ostream& out,
                const std::string& comment) {

  XCDFFile f;
  if (infiles.size() == 0) {
    f.Open(std::cin);
  } else if (infiles.size() == 1) {
    f.Open(infiles[0], "r");
  } else {
    std::cerr << "Only one input file is allowed for add-comment."
                                              " Quitting" << std::endl;
    exit(1);
  }

  XCDFFile outFile(out);

  // Get the names of all the fields
  std::set<std::string> fields;
  GetFieldNamesVisitor getFieldNamesVisitor(fields);
  f.ApplyFieldVisitor(getFieldNamesVisitor);

  // Load the fields into the buffer for copying
  FieldCopyBuffer buf(outFile);
  SelectFieldVisitor selectFieldVisitor(f, fields, buf);
  f.ApplyFieldVisitor(selectFieldVisitor);

  CopyAliases(outFile, f);
  while (f.Read()) {

    // Copy the data if true
    buf.CopyData();
    outFile.Write();
  }

  CopyComments(outFile, f);
  outFile.AddComment(comment);
  CopyAliases(outFile, f);
  outFile.Close();
}

void Comments(std::vector<std::string>& infiles,
              std::ostream& out) {

  XCDFFile f;
  for (unsigned i = 0; i <= infiles.size(); ++i) {

    if (i == infiles.size()) {
      if (infiles.size() == 0) {
        //read from stdin
        f.Open(std::cin);
      } else {
        continue;
      }
    } else {
      f.Open(infiles[i], "r");
    }

    f.LoadComments();
    for (std::vector<std::string>::const_iterator
                                   it = f.CommentsBegin();
                                   it != f.CommentsEnd(); ++it) {

      out << *it << std::endl;
    }
  }
}

template <typename Histogram, typename FillPolicy>
void FillHistogram(std::vector<std::string>& infiles,
                   Histogram& h, FillPolicy& fill) {

  XCDFFile f;
  for (unsigned i = 0; i <= infiles.size(); ++i) {

    if (i == infiles.size()) {
      if (infiles.size() == 0) {
        //read from stdin
        f.Open(std::cin);
      } else {
        continue;
      }
    } else {
      f.Open(infiles[i], "r");
    }

    fill.Fill(h, f);
  }
}

void CheckRange(std::vector<std::string>& infiles,
                RangeChecker& rc) {

  XCDFFile f;
  for (unsigned i = 0; i <= infiles.size(); ++i) {

    if (i == infiles.size()) {
      if (infiles.size() == 0) {
        XCDFFatal("Cannot fill variable range histogram from stdin")
      } else {
        continue;
      }
    } else {
      f.Open(infiles[i], "r");
    }

    rc.Fill(f);
  }
}

template <typename T>
bool Extract(std::string& s, T& out) {

  std::stringstream ss(s);
  ss >> out;
  return ss.fail();
}

void FixBins(double& min, double& max, const unsigned nbins) {

  // If all the same value, we still need max to be larger than min
  if (max < min*(1+1e-15)) {
    max = min + 1;
  } else {
    // Increase max by 1 bin to avoid cutting off upper value
    max = max + (max - min) / nbins;
  }
}

void ProcessExpression(const std::string& exp,
                       std::vector<std::string>& args) {

  // Remove whitespace, then replace the valid comma delimiters
  // with newlines.  Commas within parenthesis are not delimiters.
  int pcnt = 0;
  std::string out = exp;
  for (;;) {
    size_t pos = out.find_first_of(" \n\r\t");
    if (pos == std::string::npos) {
      break;
    }
    out.erase(pos, 1);
  }
  for (std::string::iterator it = out.begin(); it != out.end(); ++it) {
    if (*it == '(') {
      ++pcnt;
    } else if (*it == ')') {
      --pcnt;
    } else if (*it == ',' && pcnt == 0) {
      *it = '\n';
    }
  }

  char* expPtr = const_cast<char*>(out.c_str());
  for (char* tok = strtok(expPtr, "\n"); tok != NULL;
                                        tok = strtok(NULL, "\n")) {
    args.push_back(std::string(tok));
  }
}

void CreateHistogram(std::vector<std::string>& infiles,
                     std::string& exp) {

  // Parse CSV expression
  std::vector<std::string> args;
  ProcessExpression(exp, args);

  if (!(args.size() == 2 || args.size() == 3 ||
        args.size() == 4 || args.size() == 5)) {
    std::cerr << "Invalid histogram args: " << exp << std::endl;
    return;
  }

  unsigned nbins;
  double min, max;
  std::string expr;
  std::string weightExpr = "1.";
  bool fail = false;
  fail |= Extract(args[0], nbins);
  if (args.size() == 4 || args.size() == 5) {
    fail |= Extract(args[1], min);
    fail |= Extract(args[2], max);
    if (fail) {
      std::cerr << "Invalid histogram args: " << exp << std::endl;
      return;
    }
    if (nbins == 0) {
      std::cerr << "Number of bins must be greater than zero" << std::endl;
      return;
    }
    if (min > max) {
      std::cerr << "Histogram range min must be less than max" << std::endl;
      return;
    }
    expr = args[3];
    if (args.size() == 5) {
      weightExpr = args[4];
    }
  } else {
    expr = args[1];
    if (args.size() == 3) {
      weightExpr = args[2];
    }
    if (fail) {
      std::cerr << "Invalid histogram args: " << exp << std::endl;
      return;
    }
    RangeChecker rc(expr);
    CheckRange(infiles, rc);
    min = rc.GetMin();
    max = rc.GetMax();

    FixBins(min, max, nbins);
  }

  Histogram1D h(nbins, min, max);
  Filler1D fill(expr, weightExpr);
  FillHistogram(infiles, h, fill);
  std::cout << h;
}

void CreateHistogram2D(std::vector<std::string>& infiles,
                       std::string& exp) {

  // Parse CSV expression
  std::vector<std::string> args;
  ProcessExpression(exp, args);

  if (!(args.size() == 4 || args.size() == 5 ||
        args.size() == 8 || args.size() == 9)) {
    std::cerr << "Invalid histogram args: " << exp << std::endl;
    return;
  }

  unsigned nbinsX, nbinsY;
  double minX, maxX, minY, maxY;
  std::string exprX, exprY;
  std::string weightExpr = "1.";
  bool fail = false;
  fail |= Extract(args[0], nbinsX);
  if (args.size() == 8 || args.size() == 9) {
    fail |= Extract(args[1], minX);
    fail |= Extract(args[2], maxX);
    exprX = args[3];
    fail |= Extract(args[4], nbinsY);
    fail |= Extract(args[5], minY);
    fail |= Extract(args[6], maxY);
    exprY = args[7];
    if (args.size() == 9) {
      weightExpr = args[8];
    }
    if (fail) {
      std::cerr << "Invalid histogram args: " << exp << std::endl;
      return;
    }
    if (nbinsX == 0 || nbinsY == 0) {
      std::cerr << "Number of bins must be greater than zero" << std::endl;
      return;
    }
    if (minX > maxX || minY > maxY) {
      std::cerr << "Histogram range min must be less than max" << std::endl;
      return;
    }
  } else {
    exprX = args[1];
    fail |= Extract(args[2], nbinsY);
    exprY = args[3];
    if (args.size() == 5) {
      weightExpr = args[4];
    }
    if (fail) {
      std::cerr << "Invalid histogram args: " << exp << std::endl;
      return;
    }
    if (nbinsX == 0 || nbinsY == 0) {
      std::cerr << "Number of bins must be greater than zero" << std::endl;
      return;
    }
    std::vector<std::string> exprs;
    exprs.push_back(exprX);
    exprs.push_back(exprY);
    RangeChecker rc(exprs);
    CheckRange(infiles, rc);
    minX = rc.GetMin(0);
    maxX = rc.GetMax(0);
    minY = rc.GetMin(1);
    maxY = rc.GetMax(1);

    FixBins(minX, maxX, nbinsX);
    FixBins(minY, maxY, nbinsY);
  }

  Histogram2D h(nbinsX, minX, maxX, nbinsY, minY, maxY);
  Filler2D fill(exprX, exprY, weightExpr);
  FillHistogram(infiles, h, fill);
  std::cout << h;
}

void Paste(std::vector<std::string>& infiles,
           std::ostream& out,
           std::string& copyFile,
           std::string& concatArgs,
           std::string& delimeter) {

  char& del = delimeter[0];
  XCDFFile outFile(out);
  outFile.AddComment(concatArgs);

  FieldCopyBuffer buf(outFile);

  // Do we have an existing XCDF file to paste to?
  XCDFFile f;
  if (copyFile.compare("")) {

    // OK, prepare to copy it
    f.Open(copyFile, "r");

    // Get the names of all the fields
    std::set<std::string> fields;
    GetFieldNamesVisitor getFieldNamesVisitor(fields);
    f.ApplyFieldVisitor(getFieldNamesVisitor);

    // Load the fields into the buffer for copying
    SelectFieldVisitor selectFieldVisitor(f, fields, buf);
    f.ApplyFieldVisitor(selectFieldVisitor);
  }

  // Open the CSV input
  std::ifstream fileStream;
  std::istream* currentInputStream = &fileStream;

  if (infiles.size() == 0) {
    //read from stdin
    currentInputStream = &std::cin;
  } else {
    fileStream.open(infiles[0].c_str());
  }

  // Allocate the new fields
  CSVInputHandler csvIn(outFile, *currentInputStream, del);

  if (f.IsOpen()) {
    CopyAliases(outFile, f);
  }

  while(csvIn.CopyLine()) {
    if (f.IsOpen()) {
      if (!f.Read()) {
        XCDFWarn("Input file " << infiles[0] <<
                    " has fewer entries than text file.  Truncating.");
        break;
      }
      buf.CopyData();
    }

    outFile.Write();
  }

  if (fileStream.is_open()) {
    fileStream.close();
  }

  if (f.IsOpen()) {
    if (f.Read()) {
      XCDFWarn("Input text file has fewer entries than " <<
                                             copyFile << " Truncating.");
    }
    CopyComments(outFile, f);
    CopyAliases(outFile, f);
    f.Close();
  }
  outFile.Close();
}

void PrintVersion() {

  std::cout << "\n XCDF version "
            << xcdf::get_version()
            << std::endl;

}

void PrintUsage() {

  std::cout << "\n" <<

    "Usage: xcdf [verb] {infiles}\n\n" <<

    "    verb:    Description\n" <<
    "    ----     -----------\n\n"

    "    version  Print XCDF version information and exit.\n\n" <<

    "    info     Print descriptions of each field in the file.\n\n" <<

    "    dump     Output data event-by-event in a human-readable format.\n\n" <<

    "    count    {-e expression} Count the number of events in the file.\n" <<
    "             If optional expression is supplied, count only the\n" <<
    "             events that satisfy the expression.\n\n" <<

    "    csv      Output data into comma-separated-value format.\n\n" <<

    "    check    Check if input is a valid XCDF file and check internal\n" <<
    "             data checksums\n\n" <<

    "    select-fields \"field1, field2, ...\" {-o outfile} {infiles}:\n\n" <<

    "                    Copy the given fields and write the result to a\n" <<
    "                    new XCDF file at the path specified by\n" <<
    "                    {-o outfile}, or stdout if outfile is unspecified.\n" <<
    "                    A wildcard \'*\' character is allowed in matching\n" <<
    "                    field names.\n\n" <<

    "    select \"boolean expression\" {-o outfile} {infiles}:\n\n" <<

    "                    Copy events satisfying the given boolean\n" <<
    "                    expression into a new XCDF file at the path\n" <<
    "                    specified by {-o outfile}, or stdout if outfile\n" <<
    "                    is unspecified. The expression is of the form\n" <<
    "                    e.g.: \"field1 == 0\" to select all events\n" <<
    "                    where the value of field1 is zero.  The variable\n" <<
    "                    \"currentEventNumber\" refers to the current\n" <<
    "                    event in the file.\n\n" <<

    "    paste {-d delimeter} {-c existingfile} {-o outfile} {infile}:\n\n" <<

    "                    Copy events in CSV format from infile (or stdin,\n" <<
    "                    if unspecified) into outfile (or stdout if unspecified).\n" <<
    "                    If an existing XCDF file is specified with -c, the\n" <<
    "                    fields are added to the existing file. A delimeters can be \n" << 
    "                    specified but will default to commas if unspecified. \n\n" <<

    "    recover {-o outfile} {infiles} Recover a corrupt XCDF file.\n\n" <<

    "    add-alias name \"expression\" {-o outfile} {infiles}:\n\n" <<

    "                    Add an alias to \"infile\" consisting of a numerical\n" <<
    "                    expression.  The expression may contain fields, e.g.\n" <<
    "                    \'xcdf add-alias myAlias \"abs(field1)\" myFile.xcd\n" <<
    "                    creates the alias \'myAlias\' that contains the absolute\n" <<
    "                    value of XCDF field \'field1\'. \'myAlias\' may then be\n" <<
    "                    selections and other expressions.  If an output file is not\n" <<
    "                    specified, the alias is added in-place to the existing\n" <<
    "                    file if possible.  If an output file is specified,\n" <<
    "                    adding an alias with the same name as an existing alias\n" <<
    "                    replaces that alias with the new expression.  Note that\n" <<
    "                    aliases added in-place are not available when reading an\n" <<
    "                    XCDF file using a stream or pipe.\n\n" <<

    "    remove-alias name {-o outfile} {infiles}:\n\n" <<

    "                    Remove an alias from \"infile\".  If an output file is\n" <<
    "                    not specified, the removal is done in-place if possible.\n" <<
    "                    Only aliases added in-place may be removed in-place.\n\n" <<

    "    histogram \"histogram expression\" {infiles}:\n\n" <<
    "                    Create a histogram from the selected files according to\n" <<
    "                    the specified expression.  Valid expressions are of the form\n" <<
    "                    \"nbins, min, max, expr\" or \"nbins, expr\", dynamically\n" <<
    "                    determining the min and max.\n" <<
    "                    \"expr\" is of the form e.g. \"fieldName*3.14159\".\n" <<
    "                    An optional expression may be appended\n" <<
    "                    to weight the entry, e.g. \"100, 0, 1, field1, field2\" would\n" <<
    "                    create a histogram of field1 with 100 bins from 0 to 1,\n" <<
    "                    weighting each entry by the value of field2.\n\n" <<

    "    histogram2d \"histogram expression\" {infiles}:\n\n" <<
    "                    Create a 2D histogram from the selected files according to\n" <<
    "                    the specified expression.  Valid expressions are of the form\n" <<
    "                    \"nbinsX, minX, maxX, exprX, nbinsY, minY, maxY, exprY\" or\n" <<
    "                    \"nbinsX, exprX, nbinsY, exprY\", dynamically determining min\n" <<
    "                    and max.  An optional expression may be appended to weight the\n" <<
    "                    entry.\n\n" <<

    "    comments {infiles} Display all comments from an XCDF file\n\n" <<

    "    add-comment \"comment\" {-o outfile} {infiles} Add comment to an XCDF file\n\n" <<

    "    remove-comments {-o outfile} {infiles} Remove all comments from an XCDF file\n\n" <<

    "    compare file1 file2 Compare the contents of file1 and file2\n\n";

  std::cout << "\n\n";
  std::cout <<
    "  Note: if input/output file(s) are not specified, they are\n" <<
    "  read/written from/to stdin/stdout.\n\n" <<
    "  Multiple input files are allowed.\n";
}

int do_main(int argc, char** argv) {

  if (argc < 2) {
    PrintUsage();
    exit(1);
  }

  std::string concatArgs = "Arguments: ";
  for (int i = 0; i < argc; ++i) {
    concatArgs += argv[i];
    concatArgs += " ";
  }

  const std::string verb(argv[1]);

  std::string exp = "";
  std::string exp2 = "";
  std::ostream* outstream = &std::cout;
  std::ofstream fout;
  std::vector<std::string> infiles;
  std::string copyFile = "";
  std::string delimeter = ",";
  int currentArg = 2;

  if (!verb.compare("count")) {

    if (currentArg < argc) {
      std::string out(argv[currentArg]);
      if (!out.compare("-e")) {

        if (++currentArg == argc) {
          PrintUsage();
          exit(1);
        }

        // Use the user-defined cut expression to count
        exp = std::string(argv[currentArg++]);
      }
    }
  }

  if (!verb.compare("recover") || 
      !verb.compare("remove-comments")) {

    if (currentArg < argc) {

      std::string out(argv[currentArg]);
      if (!out.compare("-o")) {

        if (++currentArg == argc) {
          PrintUsage();
          exit(1);
        }

        fout.open(argv[currentArg++]);
        outstream = &fout;
      }
    }
  }

  if (!verb.compare("histogram") ||
      !verb.compare("histogram2d")) {

    if (argc < 3) {
      PrintUsage();
      exit(1);
    }

    exp = std::string(argv[currentArg++]);
  }

  if (!verb.compare("select") ||
      !verb.compare("select-fields") ||
      !verb.compare("add-comment") ||
      !verb.compare("remove-alias")) {

    if (argc < 3) {
      PrintUsage();
      exit(1);
    }
    exp = std::string(argv[currentArg++]);

    if (currentArg < argc) {

      std::string out(argv[currentArg]);
      if (!out.compare("-o")) {

        if (++currentArg == argc) {
          PrintUsage();
          exit(1);
        }

        fout.open(argv[currentArg++]);
        outstream = &fout;
      }
    }
  }

  if (!verb.compare("add-alias")) {

    if (argc < 4) {
      PrintUsage();
      exit(1);
    }
    exp = std::string(argv[currentArg++]);
    exp2 = std::string(argv[currentArg++]);

    if (currentArg < argc) {

      std::string out(argv[currentArg]);
      if (!out.compare("-o")) {

        if (++currentArg == argc) {
          PrintUsage();
          exit(1);
        }

        fout.open(argv[currentArg++]);
        outstream = &fout;
      }
    }
  }

  if (!verb.compare("paste")) {
    if (currentArg < argc) {

      std::string out(argv[currentArg]);
      if (!out.compare("-d")) {

        if(++currentArg == argc) {
          PrintUsage();
          exit(1);
        }

      delimeter = std::string(argv[currentArg++]);
      outstream = &fout;
      }
    }

    if (currentArg < argc) {

      std::string out(argv[currentArg]);
      if (!out.compare("-c")) {

        if (++currentArg == argc) {
          PrintUsage();
          exit(1);
        }

        copyFile = std::string(argv[currentArg++]);
        outstream = &fout;
      }
    }

    if (currentArg < argc) {

      std::string out(argv[currentArg]);
      if (!out.compare("-o")) {

        if (++currentArg == argc) {
          PrintUsage();
          exit(1);
        }

        fout.open(argv[currentArg++]);
        outstream = &fout;
      }
    }
  }

  while (currentArg < argc) {
    infiles.push_back(std::string(argv[currentArg++]));
  }

  if (!verb.compare("info")) {
    Info(infiles);
  }

  else if (!verb.compare("dump")) {
    Dump(infiles);
  }

  else if (!verb.compare("recover")) {
    Recover(infiles, *outstream);
  }

  else if (!verb.compare("add-alias")) {
    AddAlias(infiles, *outstream, exp, exp2);
  }

  else if (!verb.compare("remove-alias")) {
    RemoveAlias(infiles, *outstream, exp);
  }

  else if (!verb.compare("count")) {
    Count(infiles, exp);
  }

  else if (!verb.compare("csv")) {
    CSV(infiles);
  }

  else if (!verb.compare("check")) {
    Check(infiles);
  }

  else if (!verb.compare("remove-comments")) {
    RemoveComments(infiles, *outstream);
  }

  else if (!verb.compare("add-comment")) {
    AddComment(infiles, *outstream, exp);
  }

  else if (!verb.compare("comments")) {
    Comments(infiles, *outstream);
  }

  else if (!verb.compare("select-fields")) {
    SelectFields(infiles, *outstream, exp, concatArgs);
  }

  else if (!verb.compare("select")) {
    Select(infiles, *outstream, exp, concatArgs);
  }

  else if (!verb.compare("paste")) {
    if (infiles.size() > 1) {
      PrintUsage();
      exit(0);
    }
    Paste(infiles, *outstream, copyFile, concatArgs, delimeter);
  }

  else if (!verb.compare("version")) {
    PrintVersion();
  }

  else if (!verb.compare("histogram")) {
    CreateHistogram(infiles, exp);
  }

  else if (!verb.compare("histogram2d")) {
    CreateHistogram2D(infiles, exp);
  }

  else if (!verb.compare("compare")) {
    if (infiles.size() != 2) {
      PrintUsage();
      exit(1);
    }
    Compare(infiles[0], infiles[1]);
  }

  else {
    PrintUsage();
    exit(1);
  }

  return 0;
}

int main(int argc, char** argv) {
  try {
    return do_main(argc, argv);
  } catch (XCDFException& e) {
    std::cout << "Caught XCDFException: " << e.GetMessage() << std::endl;
    return -1;
  }
}
