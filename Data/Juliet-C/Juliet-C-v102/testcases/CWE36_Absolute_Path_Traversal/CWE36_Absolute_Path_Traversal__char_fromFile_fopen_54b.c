/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE36_Absolute_Path_Traversal__char_fromFile_fopen_54b.c
Label Definition File: CWE36_Absolute_Path_Traversal.label.xml
Template File: sources-sink-54b.tmpl.c
*/
/*
 * @description
 * CWE: 36 Absolute Path Traversal
 * BadSource: fromFile Read input from a file
 * GoodSource: Full path and file name
 * Sink: fopen
 *    BadSink :
 * Flow Variant: 54 Data flow: data passed as an argument from one function through three others to a fifth; all five functions are in different source files
 *
 * */

#include "std_testcase.h"

#ifdef _WIN32
# define FOPEN fopen
#else
/* fopen is used on unix-based OSs */
# define FOPEN fopen
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
void CWE36_Absolute_Path_Traversal__char_fromFile_fopen_54c_bad_sink(char * data);

void CWE36_Absolute_Path_Traversal__char_fromFile_fopen_54b_bad_sink(char * data)
{
    CWE36_Absolute_Path_Traversal__char_fromFile_fopen_54c_bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declaration */
void CWE36_Absolute_Path_Traversal__char_fromFile_fopen_54c_goodG2B_sink(char * data);

/* goodG2B uses the GoodSource with the BadSink */
void CWE36_Absolute_Path_Traversal__char_fromFile_fopen_54b_goodG2B_sink(char * data)
{
    CWE36_Absolute_Path_Traversal__char_fromFile_fopen_54c_goodG2B_sink(data);
}

#endif /* OMITGOOD */
