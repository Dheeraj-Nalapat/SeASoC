/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__w32CreateFile_no_close_53b.c
Label Definition File: CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__w32CreateFile_no_close.label.xml
Template File: source-sinks-53b.tmpl.c
*/
/*
 * @description
 * CWE: 772 Missing Release of Resource after Effective Lifetime
 * BadSource:  Open a file using CreateFile()
 * Sinks:
 *    GoodSink: Close the file using CloseHandle()
 *    BadSink : Do not close file
 * Flow Variant: 53 Data flow: data passed as an argument from one function through two others to a fourth; all four functions are in different source files
 *
 * */

#include "std_testcase.h"

#include <windows.h>

#ifndef OMITBAD

/* bad function declaration */
void CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__w32CreateFile_no_close_53c_bad_sink(HANDLE data);

void CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__w32CreateFile_no_close_53b_bad_sink(HANDLE data)
{
    CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__w32CreateFile_no_close_53c_bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodB2G uses the BadSource with the GoodSink */
void CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__w32CreateFile_no_close_53c_goodB2G_sink(HANDLE data);

void CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__w32CreateFile_no_close_53b_goodB2G_sink(HANDLE data)
{
    CWE772_Missing_Release_of_Resource_after_Effective_Lifetime__w32CreateFile_no_close_53c_goodB2G_sink(data);
}

#endif /* OMITGOOD */
