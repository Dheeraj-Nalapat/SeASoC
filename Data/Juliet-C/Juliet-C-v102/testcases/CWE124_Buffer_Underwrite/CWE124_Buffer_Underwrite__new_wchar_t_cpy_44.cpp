/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE124_Buffer_Underwrite__new_wchar_t_cpy_44.cpp
Label Definition File: CWE124_Buffer_Underwrite__new.label.xml
Template File: sources-sink-44.tmpl.cpp
*/
/*
 * @description
 * CWE: 124 Buffer Underwrite
 * BadSource:  Set data pointer to before the allocated memory buffer
 * GoodSource: Set data pointer to the allocated memory buffer
 * Sinks: cpy
 *    BadSink : Copy string to data using wcscpy
 * Flow Variant: 44 Data/control flow: data passed as an argument from one function to a function in the same source file called via a function pointer
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

namespace CWE124_Buffer_Underwrite__new_wchar_t_cpy_44
{

#ifndef OMITBAD

static void bad_sink(wchar_t * data)
{
    {
        wchar_t src[100];
        wmemset(src, L'C', 100-1); /* fill with 'C's */
        src[100-1] = L'\0'; /* null terminate */
        /* POTENTIAL FLAW: Possibly copying data to memory before the destination buffer */
        wcscpy(data, src);
        printWLine(data);
        /* INCIDENTAL CWE-401: Memory Leak - data may not point to location
           returned by new [] so can't safely call delete [] on it */
    }
}

void bad()
{
    wchar_t * data;
    /* define a function pointer */
    void (*func_ptr) (wchar_t *) = bad_sink;
    data = NULL;
    {
        wchar_t * data_buf = new wchar_t[100];
        wmemset(data_buf, L'A', 100-1);
        data_buf[100-1] = L'\0';
        /* FLAW: Set data pointer to before the allocated memory buffer */
        data = data_buf - 8;
    }
    /* use the function pointer */
    func_ptr(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B_sink(wchar_t * data)
{
    {
        wchar_t src[100];
        wmemset(src, L'C', 100-1); /* fill with 'C's */
        src[100-1] = L'\0'; /* null terminate */
        /* POTENTIAL FLAW: Possibly copying data to memory before the destination buffer */
        wcscpy(data, src);
        printWLine(data);
        /* INCIDENTAL CWE-401: Memory Leak - data may not point to location
           returned by new [] so can't safely call delete [] on it */
    }
}

static void goodG2B()
{
    wchar_t * data;
    void (*func_ptr) (wchar_t *) = goodG2B_sink;
    data = NULL;
    {
        wchar_t * data_buf = new wchar_t[100];
        wmemset(data_buf, L'A', 100-1);
        data_buf[100-1] = L'\0';
        /* FIX: Set data pointer to the allocated memory buffer */
        data = data_buf;
    }
    func_ptr(data);
}

void good()
{
    goodG2B();
}

#endif /* OMITGOOD */

} // close namespace

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */

#ifdef INCLUDEMAIN

using namespace CWE124_Buffer_Underwrite__new_wchar_t_cpy_44; // so that we can use good and bad easily

int main(int argc, char * argv[])
{
    /* seed randomness */
    srand( (unsigned)time(NULL) );
#ifndef OMITGOOD
    printLine("Calling good()...");
    good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
