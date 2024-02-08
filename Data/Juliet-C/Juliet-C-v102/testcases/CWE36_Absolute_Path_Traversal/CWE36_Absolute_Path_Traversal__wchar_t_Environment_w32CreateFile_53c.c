/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE36_Absolute_Path_Traversal__wchar_t_Environment_w32CreateFile_53c.c
Label Definition File: CWE36_Absolute_Path_Traversal.label.xml
Template File: sources-sink-53c.tmpl.c
*/
/*
 * @description
 * CWE: 36 Absolute Path Traversal
 * BadSource: Environment Read input from an environment variable
 * GoodSource: Full path and file name
 * Sink: w32CreateFile
 *    BadSink :
 * Flow Variant: 53 Data flow: data passed as an argument from one function through two others to a fourth; all four functions are in different source files
 *
 * */

#include "std_testcase.h"

#define ENV_VARIABLE L"ADD"

#ifdef _WIN32
# define GETENV _wgetenv
#else
# define GETENV wgetenv
#endif

#include <windows.h>

/* all the sinks are the same, we just want to know where the hit originated if a tool flags one */

#ifndef OMITBAD

/* bad function declaration */
void CWE36_Absolute_Path_Traversal__wchar_t_Environment_w32CreateFile_53d_bad_sink(wchar_t * data);

void CWE36_Absolute_Path_Traversal__wchar_t_Environment_w32CreateFile_53c_bad_sink(wchar_t * data)
{
    CWE36_Absolute_Path_Traversal__wchar_t_Environment_w32CreateFile_53d_bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declaration */
void CWE36_Absolute_Path_Traversal__wchar_t_Environment_w32CreateFile_53d_goodG2B_sink(wchar_t * data);

/* goodG2B uses the GoodSource with the BadSink */
void CWE36_Absolute_Path_Traversal__wchar_t_Environment_w32CreateFile_53c_goodG2B_sink(wchar_t * data)
{
    CWE36_Absolute_Path_Traversal__wchar_t_Environment_w32CreateFile_53d_goodG2B_sink(data);
}

#endif /* OMITGOOD */
