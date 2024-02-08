/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE401_Memory_Leak__strdup_char_65a.c
Label Definition File: CWE401_Memory_Leak__strdup.label.xml
Template File: sources-sinks-65a.tmpl.c
*/
/*
 * @description
 * CWE: 401 Memory Leak
 * BadSource:  Allocate data using strdup()
 * GoodSource: Allocate data on the stack
 * Sinks:
 *    GoodSink: call free() on data
 *    BadSink : no deallocation of data
 * Flow Variant: 65 Data/control flow: data passed as an argument from one function to a function in a different source file called via a function pointer
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

#ifndef OMITBAD

/* bad function declaration */
void CWE401_Memory_Leak__strdup_char_65b_bad_sink(char * data);

void CWE401_Memory_Leak__strdup_char_65_bad()
{
    char * data;
    /* define a function pointer */
    void (*func_ptr) (char *) = CWE401_Memory_Leak__strdup_char_65b_bad_sink;
    data = NULL;
    {
        char mystring[] = "mystring";
        /* FLAW: Allocate memory from the heap using a function that requires free() for deallocation */
        data = strdup(mystring);
        /* Use data */
        printLine(data);
    }
    /* use the function pointer */
    func_ptr(data);
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B uses the GoodSource with the BadSink */
void CWE401_Memory_Leak__strdup_char_65b_goodG2B_sink(char * data);

static void goodG2B()
{
    char * data;
    void (*func_ptr) (char *) = CWE401_Memory_Leak__strdup_char_65b_goodG2B_sink;
    data = NULL;
    {
        /* FIX: Allocate memory from the stack */
        char data_goodbuf[100];
        data = data_goodbuf;
        /* Initialize then use data */
        strcpy(data, "a string");
        printLine(data);
    }
    func_ptr(data);
}

/* goodB2G uses the BadSource with the GoodSink */
void CWE401_Memory_Leak__strdup_char_65b_goodB2G_sink(char * data);

static void goodB2G()
{
    char * data;
    void (*func_ptr) (char *) = CWE401_Memory_Leak__strdup_char_65b_goodB2G_sink;
    data = NULL;
    {
        char mystring[] = "mystring";
        /* FLAW: Allocate memory from the heap using a function that requires free() for deallocation */
        data = strdup(mystring);
        /* Use data */
        printLine(data);
    }
    func_ptr(data);
}

void CWE401_Memory_Leak__strdup_char_65_good()
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
    CWE401_Memory_Leak__strdup_char_65_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE401_Memory_Leak__strdup_char_65_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
