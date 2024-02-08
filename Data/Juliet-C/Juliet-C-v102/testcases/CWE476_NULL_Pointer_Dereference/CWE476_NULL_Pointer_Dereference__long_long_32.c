/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE476_NULL_Pointer_Dereference__long_long_32.c
Label Definition File: CWE476_NULL_Pointer_Dereference.label.xml
Template File: sources-sink-32.tmpl.c
*/
/*
 * @description
 * CWE: 476 NULL Pointer Dereference
 * BadSource:  Set data to NULL
 * GoodSource: Initialize data
 * Sink:
 *    BadSink : Print data
 * Flow Variant: 32 Data flow using two pointers to the same value within the same function
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifndef OMITBAD

void CWE476_NULL_Pointer_Dereference__long_long_32_bad()
{
    long long * data;
    long long * *data_ptr1 = &data;
    long long * *data_ptr2 = &data;
    /* Initialize data */
    data = NULL;
    {
        long long * data = *data_ptr1;
        /* FLAW: Set data to NULL - it will be used in the sink */
        data = NULL;
        *data_ptr1 = data;
    }
    {
        long long * data = *data_ptr2;
        /* POTENTIAL FLAW: Attempt to use data, which may be NULL */
        printLongLongLine(*data);
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B()
{
    long long * data;
    long long * *data_ptr1 = &data;
    long long * *data_ptr2 = &data;
    /* Initialize data */
    data = NULL;
    {
        long long * data = *data_ptr1;
        /* FIX: Initialize data - it will be used in the sink */
        {
            long long tmp = 5L;
            data = &tmp;
        }
        *data_ptr1 = data;
    }
    {
        long long * data = *data_ptr2;
        /* POTENTIAL FLAW: Attempt to use data, which may be NULL */
        printLongLongLine(*data);
    }
}

void CWE476_NULL_Pointer_Dereference__long_long_32_good()
{
    goodG2B();
}

#endif /* OMITGOOD */

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */
#ifdef INCLUDEMAIN

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    CWE476_NULL_Pointer_Dereference__long_long_32_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE476_NULL_Pointer_Dereference__long_long_32_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
