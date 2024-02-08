/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE124_Buffer_Underwrite__wchar_t_declare_ncpy_52b.c
Label Definition File: CWE124_Buffer_Underwrite.stack.label.xml
Template File: sources-sink-52b.tmpl.c
*/
/*
 * @description
 * CWE: 124 Buffer Underwrite
 * BadSource:  Set data pointer to before the allocated memory buffer
 * GoodSource: Set data pointer to the allocated memory buffer
 * Sink: ncpy
 *    BadSink : Copy string to data using wcsncpy
 * Flow Variant: 52 Data flow: data passed as an argument from one function to another to another in three different source files
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

/* all the sinks are the same, we just want to know where the hit originated if a tool flags one */

#ifndef OMITBAD

/* bad function declaration */
void CWE124_Buffer_Underwrite__wchar_t_declare_ncpy_52c_bad_sink(wchar_t * data);

void CWE124_Buffer_Underwrite__wchar_t_declare_ncpy_52b_bad_sink(wchar_t * data)
{
    CWE124_Buffer_Underwrite__wchar_t_declare_ncpy_52c_bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* good function declaration */
void CWE124_Buffer_Underwrite__wchar_t_declare_ncpy_52c_goodG2B_sink(wchar_t * data);

/* goodG2B uses the GoodSource with the BadSink */
void CWE124_Buffer_Underwrite__wchar_t_declare_ncpy_52b_goodG2B_sink(wchar_t * data)
{
    CWE124_Buffer_Underwrite__wchar_t_declare_ncpy_52c_goodG2B_sink(data);
}

#endif /* OMITGOOD */
