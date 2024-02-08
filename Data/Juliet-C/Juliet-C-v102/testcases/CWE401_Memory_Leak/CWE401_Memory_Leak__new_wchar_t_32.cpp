/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE401_Memory_Leak__new_wchar_t_32.cpp
Label Definition File: CWE401_Memory_Leak__new.label.xml
Template File: sources-sinks-32.tmpl.cpp
*/
/*
 * @description
 * CWE: 401 Memory Leak
 * BadSource:  Allocate data using new
 * GoodSource: Allocate data on the stack
 * Sinks:
 *    GoodSink: call delete on data
 *    BadSink : no deallocation of data
 * Flow Variant: 32 Data flow using two pointers to the same value within the same function
 *
 * */

#include "std_testcase.h"

namespace CWE401_Memory_Leak__new_wchar_t_32
{

#ifndef OMITBAD

void bad()
{
    wchar_t * data;
    wchar_t * *data_ptr1 = &data;
    wchar_t * *data_ptr2 = &data;
    data = NULL;
    {
        wchar_t * data = *data_ptr1;
        /* POTENTIAL FLAW: Allocate memory on the heap */
        data = new wchar_t;
        /* Initialize and make use of data */
        *data = L'A';
        printHexCharLine((char)*data);
        *data_ptr1 = data;
    }
    {
        wchar_t * data = *data_ptr2;
        /* POTENTIAL FLAW: No deallocation */
        ; /* empty statement needed for some flow variants */
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B()
{
    wchar_t * data;
    wchar_t * *data_ptr1 = &data;
    wchar_t * *data_ptr2 = &data;
    data = NULL;
    {
        wchar_t * data = *data_ptr1;
        /* FIX: Use memory allocated on the stack */
        wchar_t data_good;
        data = &data_good;
        /* Initialize and make use of data */
        *data = L'A';
        printHexCharLine((char)*data);
        *data_ptr1 = data;
    }
    {
        wchar_t * data = *data_ptr2;
        /* POTENTIAL FLAW: No deallocation */
        ; /* empty statement needed for some flow variants */
    }
}

/* goodB2G() uses the BadSource with the GoodSink */
static void goodB2G()
{
    wchar_t * data;
    wchar_t * *data_ptr1 = &data;
    wchar_t * *data_ptr2 = &data;
    data = NULL;
    {
        wchar_t * data = *data_ptr1;
        /* POTENTIAL FLAW: Allocate memory on the heap */
        data = new wchar_t;
        /* Initialize and make use of data */
        *data = L'A';
        printHexCharLine((char)*data);
        *data_ptr1 = data;
    }
    {
        wchar_t * data = *data_ptr2;
        /* FIX: Deallocate memory */
        delete data;
    }
}

void good()
{
    goodG2B();
    goodB2G();
}

#endif /* OMITGOOD */

} // close namespace

/* Below is the main(). It is only used when building this testcase on
   its own for testing or for building a binary to use in testing binary
   analysis tools. It is not used when compiling all the testcases as one
   application, which is how source code analysis tools are tested. */
#ifdef INCLUDEMAIN

using namespace CWE401_Memory_Leak__new_wchar_t_32; // so that we can use good and bad easily

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
