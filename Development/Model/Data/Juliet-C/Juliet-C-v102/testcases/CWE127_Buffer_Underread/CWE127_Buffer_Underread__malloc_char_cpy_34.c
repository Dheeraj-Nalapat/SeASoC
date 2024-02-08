/* TEMPLATE GENERATED TESTCASE FILE
Filename: CWE127_Buffer_Underread__malloc_char_cpy_34.c
Label Definition File: CWE127_Buffer_Underread__malloc.label.xml
Template File: sources-sink-34.tmpl.c
*/
/*
 * @description
 * CWE: 127 Buffer Under-read
 * BadSource:  Set data pointer to before the allocated memory buffer
 * GoodSource: Set data pointer to the allocated memory buffer
 * Sinks: cpy
 *    BadSink : Copy data to string using strcpy
 * Flow Variant: 34 Data flow: use of a union containing two methods of accessing the same data (within the same function)
 *
 * */

#include "std_testcase.h"

#include <wchar.h>

typedef union
{
    char * a;
    char * b;
} CWE127_Buffer_Underread__malloc_char_cpy_34_union_type;

#ifndef OMITBAD

void CWE127_Buffer_Underread__malloc_char_cpy_34_bad()
{
    char * data;
    CWE127_Buffer_Underread__malloc_char_cpy_34_union_type my_union;
    data = NULL;
    {
        char * data_buf = (char *)malloc(100*sizeof(char));
        memset(data_buf, 'A', 100-1);
        data_buf[100-1] = '\0';
        /* FLAW: Set data pointer to before the allocated memory buffer */
        data = data_buf - 8;
    }
    my_union.a = data;
    {
        char * data = my_union.b;
        {
            char dest[100];
            memset(dest, 'C', 100-1); /* fill with 'C's */
            dest[100-1] = '\0'; /* null terminate */
            /* POTENTIAL FLAW: Possibly copy from a memory location located before the source buffer */
            strcpy(dest, data);
            printLine(dest);
            /* INCIDENTAL CWE-401: Memory Leak - data may not point to location
               returned by malloc() so can't safely call free() on it */
        }
    }
}

#endif /* OMITBAD */

#ifndef OMITGOOD

/* goodG2B() uses the GoodSource with the BadSink */
static void goodG2B()
{
    char * data;
    CWE127_Buffer_Underread__malloc_char_cpy_34_union_type my_union;
    data = NULL;
    {
        char * data_buf = (char *)malloc(100*sizeof(char));
        memset(data_buf, 'A', 100-1);
        data_buf[100-1] = '\0';
        /* FIX: Set data pointer to the allocated memory buffer */
        data = data_buf;
    }
    my_union.a = data;
    {
        char * data = my_union.b;
        {
            char dest[100];
            memset(dest, 'C', 100-1); /* fill with 'C's */
            dest[100-1] = '\0'; /* null terminate */
            /* POTENTIAL FLAW: Possibly copy from a memory location located before the source buffer */
            strcpy(dest, data);
            printLine(dest);
            /* INCIDENTAL CWE-401: Memory Leak - data may not point to location
               returned by malloc() so can't safely call free() on it */
        }
    }
}

void CWE127_Buffer_Underread__malloc_char_cpy_34_good()
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
    CWE127_Buffer_Underread__malloc_char_cpy_34_good();
    printLine("Finished good()");
#endif /* OMITGOOD */
#ifndef OMITBAD
    printLine("Calling bad()...");
    CWE127_Buffer_Underread__malloc_char_cpy_34_bad();
    printLine("Finished bad()");
#endif /* OMITBAD */
    return 0;
}

#endif
