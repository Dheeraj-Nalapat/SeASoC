/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE197_Numeric_Truncation_Error__int_large_to_short_54e.c
Label Definition File: CWE197_Numeric_Truncation_Error__int.label.xml
Template File: sources-sink-54e.tmpl.c
*/
/*
 * @description
 * CWE: 197 Numeric Truncation Error
 * BadSource: large Set data to a number larger than SHRT_MAX
 * GoodSource: Less than CHAR_MAX
 * Sink: to_short
 *    BadSink : Convert data to a short
 * Flow Variant: 54 Data flow: data passed as an argument from one function through three others to a fifth; all five functions are in different source files
 *
 * */

#include "std_testcase.h"

/* all the sinks are the same, we just want to know where the hit originated if a tool flags one */

#ifndef OMITBAD

void CWE197_Numeric_Truncation_Error__int_large_to_short_54e_bad_sink(int data)
{
    {
        /* POTENTIAL FLAW: Convert data to a short, possibly causing a truncation error */
        short s = (short)data;
        printf("%hd\n", s);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE197_Numeric_Truncation_Error__int_large_to_short_54e_goodG2B_sink(int data)
{
    {
        /* POTENTIAL FLAW: Convert data to a short, possibly causing a truncation error */
        short s = (short)data;
        printf("%hd\n", s);
    }
}

#endif /* OMITGOOD */
