/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE476_NULL_Pointer_Dereference__wchar_t_41.c
Label Definition File: CWE476_NULL_Pointer_Dereference.label.xml
Template File: sources-sink-41.tmpl.c
*/
/*
 * @description
 * CWE: 476 NULL Pointer Dereference
 * BadSource:  Set data to NULL
 * GoodSource: Initialize data
 * Sink:
 *    BadSink : Print data
 * Flow Variant: 41 Data flow: data passed as an argument from one function to another in the same source file
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifndef OMITBAD

void CWE476_NULL_Pointer_Dereference__wchar_t_41_bad_sink(wchar_t * data)
{
    /* POTENTIAL FLAW: Attempt to use data, which may be NULL */
    /* printWLine() checks for NULL, so we cannot use it here */
    wprintf(L"%s\n", data);
}

void CWE476_NULL_Pointer_Dereference__wchar_t_41_bad()
{
    wchar_t * data;
    /* Initialize data */
    data = NULL;
    /* FLAW: Set data to NULL - it will be used in the sink */
    data = NULL;
    CWE476_NULL_Pointer_Dereference__wchar_t_41_bad_sink(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

void CWE476_NULL_Pointer_Dereference__wchar_t_41_goodG2B_sink(wchar_t * data)
{
    /* POTENTIAL FLAW: Attempt to use data, which may be NULL */
    /* printWLine() checks for NULL, so we cannot use it here */
    wprintf(L"%s\n", data);
}

/* goodG2B uses the GoodSource with the BadSink */
static void goodG2B()
{
    wchar_t * data;
    /* Initialize data */
    data = NULL;
    /* FIX: Initialize data - it will be used in the sink */
    data = L"Good";
    CWE476_NULL_Pointer_Dereference__wchar_t_41_goodG2B_sink(data);
}

void CWE476_NULL_Pointer_Dereference__wchar_t_41_good()
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
    CWE476_NULL_Pointer_Dereference__wchar_t_41_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE476_NULL_Pointer_Dereference__wchar_t_41_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
