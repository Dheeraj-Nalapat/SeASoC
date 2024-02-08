/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE127_Buffer_Underread__char_declare_memmove_54d.c
Label Definition File: CWE127_Buffer_Underread.stack.label.xml
Template File: sources-sink-54d.tmpl.c
*/
/*
 * @description
 * CWE: 127 Buffer Under-read
 * BadSource:  Set data pointer to before the allocated memory buffer
 * GoodSource: Set data pointer to the allocated memory buffer
 * Sink: memmove
 *    BadSink : Copy data to string using memmove
 * Flow Variant: 54 Data flow: data passed as an argument from one function through three others to a fifth; all five functions are in different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

/* all the sinks are the same, we just want to know where the hit originated if a tool flags one */

#ifndef OMITBAD

/* bad function declaration */
void CWE127_Buffer_Underread__char_declare_memmove_54e_bad_sink(char * data);

void CWE127_Buffer_Underread__char_declare_memmove_54d_bad_sink(char * data)
{
    CWE127_Buffer_Underread__char_declare_memmove_54e_bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declaration */
void CWE127_Buffer_Underread__char_declare_memmove_54e_goodG2B_sink(char * data);

/* goodG2B uses the GoodSource with the BadSink */
void CWE127_Buffer_Underread__char_declare_memmove_54d_goodG2B_sink(char * data)
{
    CWE127_Buffer_Underread__char_declare_memmove_54e_goodG2B_sink(data);
}

#endif /* OMITGOOD */
