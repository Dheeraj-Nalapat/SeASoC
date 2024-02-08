/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE401_Memory_Leak__strdup_char_32.c
Label Definition File: CWE401_Memory_Leak__strdup.label.xml
Template File: sources-sinks-32.tmpl.c
*/
/*
 * @description
 * CWE: 401 Memory Leak
 * BadSource:  Allocate data using strdup()
 * GoodSource: Allocate data on the stack
 * Sinks:
 *    GoodSink: call free() on data
 *    BadSink : no deallocation of data
 * Flow Variant: 32 Data flow using two pointers to the same value within the same function
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifndef OMITBAD

void CWE401_Memory_Leak__strdup_char_32_bad()
{
    char * data;
    char * *data_ptr1 = &data;
    char * *data_ptr2 = &data;
    data = NULL;
    {
        char * data = *data_ptr1;
        {
            char mystring[] = "mystring";
            /* FLAW: Allocate memory from the heap using a function that requires free() for deallocation */
            data = strdup(mystring);
            /* Use data */
            printLine(data);
        }
        *data_ptr1 = data;
    }
    {
        char * data = *data_ptr2;
        /* POTENTIAL FLAW: No deallocation of memory */
        /* no deallocation */
        ; /* empty statement needed for some flow variants */
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B()
{
    char * data;
    char * *data_ptr1 = &data;
    char * *data_ptr2 = &data;
    data = NULL;
    {
        char * data = *data_ptr1;
        {
            /* FIX: Allocate memory from the stack */
            char data_goodbuf[100];
            data = data_goodbuf;
            /* Initialize then use data */
            strcpy(data, "a string");
            printLine(data);
        }
        *data_ptr1 = data;
    }
    {
        char * data = *data_ptr2;
        /* POTENTIAL FLAW: No deallocation of memory */
        /* no deallocation */
        ; /* empty statement needed for some flow variants */
    }
}

/* goodB2G() uses the BadSource with the GoodSink */
static void goodB2G()
{
    char * data;
    char * *data_ptr1 = &data;
    char * *data_ptr2 = &data;
    data = NULL;
    {
        char * data = *data_ptr1;
        {
            char mystring[] = "mystring";
            /* FLAW: Allocate memory from the heap using a function that requires free() for deallocation */
            data = strdup(mystring);
            /* Use data */
            printLine(data);
        }
        *data_ptr1 = data;
    }
    {
        char * data = *data_ptr2;
        /* FIX: Deallocate memory initialized in the source */
        free(data);
    }
}

void CWE401_Memory_Leak__strdup_char_32_good()
{
    goodG2B();
    goodB2G();
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
    CWE401_Memory_Leak__strdup_char_32_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE401_Memory_Leak__strdup_char_32_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
