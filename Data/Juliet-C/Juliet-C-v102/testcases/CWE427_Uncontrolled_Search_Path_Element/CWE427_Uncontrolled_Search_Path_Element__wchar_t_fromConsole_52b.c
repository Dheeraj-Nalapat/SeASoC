/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE427_Uncontrolled_Search_Path_Element__wchar_t_fromConsole_52b.c
Label Definition File: CWE427_Uncontrolled_Search_Path_Element.label.xml
Template File: sources-sink-52b.tmpl.c
*/
/*
 * @description
 * CWE: 427 Uncontrolled Search Path Element
 * BadSource: fromConsole Read input from the console
 * GoodSource: Use a hardcoded path
 * Sink:
 *    BadSink : Set the environment variable
 * Flow Variant: 52 Data flow: data passed as an argument from one function to another to another in three different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifdef _WIN32
# define NEW_PATH L"%SystemRoot%\\system32"
# define PUTENV _wputenv
#else
# define NEW_PATH L"/bin"
# define PUTENV wputenv
#endif

/* all the sinks are the same, we just want to know where the hit originated if a tool flags one */

#ifndef OMITBAD

/* bad function declaration */
void CWE427_Uncontrolled_Search_Path_Element__wchar_t_fromConsole_52c_bad_sink(wchar_t * data);

void CWE427_Uncontrolled_Search_Path_Element__wchar_t_fromConsole_52b_bad_sink(wchar_t * data)
{
    CWE427_Uncontrolled_Search_Path_Element__wchar_t_fromConsole_52c_bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declaration */
void CWE427_Uncontrolled_Search_Path_Element__wchar_t_fromConsole_52c_goodG2B_sink(wchar_t * data);

/* goodG2B uses the GoodSource with the BadSink */
void CWE427_Uncontrolled_Search_Path_Element__wchar_t_fromConsole_52b_goodG2B_sink(wchar_t * data)
{
    CWE427_Uncontrolled_Search_Path_Element__wchar_t_fromConsole_52c_goodG2B_sink(data);
}

#endif /* OMITGOOD */
