/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE121_Stack_Based_Buffer_Overflow__dest_long_long_declare_memcpy_68b.c
Label Definition File: CWE121_Stack_Based_Buffer_Overflow__dest.label.xml
Template File: sources-sink-68b.tmpl.c
*/
/*
 * @description
 * CWE: 121 Stack Based Buffer Overflow
 * BadSource:  Set data pointer to the bad buffer
 * GoodSource: Set data pointer to the good buffer
 * Sink: memcpy
 *    BadSink : Copy long long array to data using memcpy
 * Flow Variant: 68 Data flow: data passed as a global variable from one function to another in different source files
 *
 * */

#include "std_testcase.h"

extern long long * CWE121_Stack_Based_Buffer_Overflow__dest_long_long_declare_memcpy_68_bad_data;
extern long long * CWE121_Stack_Based_Buffer_Overflow__dest_long_long_declare_memcpy_68_goodG2B_data;

/* all the sinks are the same, we just want to know where the hit originated if a tool flags one */

#ifndef OMITBAD

void CWE121_Stack_Based_Buffer_Overflow__dest_long_long_declare_memcpy_68b_bad_sink()
{
    long long * data = CWE121_Stack_Based_Buffer_Overflow__dest_long_long_declare_memcpy_68_bad_data;
    {
        long long src[100] = {0}; /* fill with 0's */
        /* POTENTIAL FLAW: Possible buffer overflow if data < 100 */
        memcpy(data, src, 100*sizeof(long long));
        printLongLongLine(data[0]);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE121_Stack_Based_Buffer_Overflow__dest_long_long_declare_memcpy_68b_goodG2B_sink()
{
    long long * data = CWE121_Stack_Based_Buffer_Overflow__dest_long_long_declare_memcpy_68_goodG2B_data;
    {
        long long src[100] = {0}; /* fill with 0's */
        /* POTENTIAL FLAW: Possible buffer overflow if data < 100 */
        memcpy(data, src, 100*sizeof(long long));
        printLongLongLine(data[0]);
    }
}

#endif /* OMITGOOD */
