/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE427_Uncontrolled_Search_Path_Element__char_fromFile_53c.c
Label Definition File: CWE427_Uncontrolled_Search_Path_Element.label.xml
Template File: sources-sink-53c.tmpl.c
*/
/*
 * @description
 * CWE: 427 Uncontrolled Search Path Element
 * BadSource: fromFile Read input from a file
 * GoodSource: Use a hardcoded path
 * Sink:
 *    BadSink : Set the environment variable
 * Flow Variant: 53 Data flow: data passed as an argument from one function through two others to a fourth; all four functions are in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifdef _WIN32
# define NEW_PATH "%SystemRoot%\\system32"
# define PUTENV _putenv
#else
# define NEW_PATH "/bin"
# define PUTENV putenv
#endif

#ifdef _WIN32
# define FOPEN fopen
#else
/* fopen is used on unix-based OSs */
# define FOPEN fopen
#endif

/* all the sinks are the same, we just want to know where the hit originated if a tool flags one */

#ifndef OMITBAD

/* bad function declaration */
void CWE427_Uncontrolled_Search_Path_Element__char_fromFile_53d_bad_sink(char * data);

void CWE427_Uncontrolled_Search_Path_Element__char_fromFile_53c_bad_sink(char * data)
{
    CWE427_Uncontrolled_Search_Path_Element__char_fromFile_53d_bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declaration */
void CWE427_Uncontrolled_Search_Path_Element__char_fromFile_53d_goodG2B_sink(char * data);

/* goodG2B uses the GoodSource with the BadSink */
void CWE427_Uncontrolled_Search_Path_Element__char_fromFile_53c_goodG2B_sink(char * data)
{
    CWE427_Uncontrolled_Search_Path_Element__char_fromFile_53d_goodG2B_sink(data);
}

#endif /* OMITGOOD */
