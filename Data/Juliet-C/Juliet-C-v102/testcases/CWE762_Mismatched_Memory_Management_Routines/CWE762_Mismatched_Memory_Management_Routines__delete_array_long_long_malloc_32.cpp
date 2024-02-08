/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE762_Mismatched_Memory_Management_Routines__delete_array_long_long_malloc_32.cpp
Label Definition File: CWE762_Mismatched_Memory_Management_Routines__delete_array.label.xml
Template File: sources-sinks-32.tmpl.cpp
*/
/*
 * @description
 * CWE: 762 Mismatched Memory Management Routines
 * BadSource: malloc Allocate data using malloc()
 * GoodSource: Allocate data using new []
 * Sinks:
 *    GoodSink: Deallocate data using free()
 *    BadSink : Deallocate data using delete []
 * Flow Variant: 32 Data flow using two pointers to the same value within the same function
 *
 * */

#include "std_testcase.h"

namespace CWE762_Mismatched_Memory_Management_Routines__delete_array_long_long_malloc_32
{

#ifndef OMITBAD

void bad()
{
    long long * data;
    long long * *data_ptr1 = &data;
    long long * *data_ptr2 = &data;
    /* Initialize data*/
    data = NULL;
    {
        long long * data = *data_ptr1;
        /* POTENTIAL FLAW: Allocate memory with a function that requires free() to free the memory */
        data = (long long *)malloc(100*sizeof(long long));
        *data_ptr1 = data;
    }
    {
        long long * data = *data_ptr2;
        /* POTENTIAL FLAW: Deallocate memory using delete [] - the source memory allocation function may
         * require a call to free() to deallocate the memory */
        delete [] data;
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
    /* Initialize data*/
    data = NULL;
    {
        long long * data = *data_ptr1;
        /* FIX: Allocate memory using new [] */
        data = new long long[100];
        *data_ptr1 = data;
    }
    {
        long long * data = *data_ptr2;
        /* POTENTIAL FLAW: Deallocate memory using delete [] - the source memory allocation function may
         * require a call to free() to deallocate the memory */
        delete [] data;
    }
}

/* goodB2G() uses the BadSource with the GoodSink */
static void goodB2G()
{
    long long * data;
    long long * *data_ptr1 = &data;
    long long * *data_ptr2 = &data;
    /* Initialize data*/
    data = NULL;
    {
        long long * data = *data_ptr1;
        /* POTENTIAL FLAW: Allocate memory with a function that requires free() to free the memory */
        data = (long long *)malloc(100*sizeof(long long));
        *data_ptr1 = data;
    }
    {
        long long * data = *data_ptr2;
        /* FIX: Free memory using free() */
        free(data);
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

using namespace CWE762_Mismatched_Memory_Management_Routines__delete_array_long_long_malloc_32; // so that we can use good and bad easily

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
