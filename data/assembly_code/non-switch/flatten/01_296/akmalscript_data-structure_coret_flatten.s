	.file	"akmalscript_data-structure_coret_flatten.c"
	.text
	.globl	_TIG_IZ_51Ly_argv
	.bss
	.align 8
	.type	_TIG_IZ_51Ly_argv, @object
	.size	_TIG_IZ_51Ly_argv, 8
_TIG_IZ_51Ly_argv:
	.zero	8
	.globl	_TIG_IZ_51Ly_argc
	.align 4
	.type	_TIG_IZ_51Ly_argc, @object
	.size	_TIG_IZ_51Ly_argc, 4
_TIG_IZ_51Ly_argc:
	.zero	4
	.globl	_TIG_IZ_51Ly_envp
	.align 8
	.type	_TIG_IZ_51Ly_envp, @object
	.size	_TIG_IZ_51Ly_envp, 8
_TIG_IZ_51Ly_envp:
	.zero	8
	.section	.rodata
.LC0:
	.string	"Position is out of bounds."
	.text
	.globl	insertAtPosition
	.type	insertAtPosition, @function
insertAtPosition:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movq	%rdi, -56(%rbp)
	movl	%esi, -60(%rbp)
	movl	%edx, -64(%rbp)
	movq	$20, -16(%rbp)
.L28:
	cmpq	$21, -16(%rbp)
	ja	.L29
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L4(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L4(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L4:
	.long	.L17-.L4
	.long	.L29-.L4
	.long	.L16-.L4
	.long	.L29-.L4
	.long	.L29-.L4
	.long	.L15-.L4
	.long	.L29-.L4
	.long	.L14-.L4
	.long	.L13-.L4
	.long	.L12-.L4
	.long	.L11-.L4
	.long	.L10-.L4
	.long	.L29-.L4
	.long	.L9-.L4
	.long	.L29-.L4
	.long	.L29-.L4
	.long	.L8-.L4
	.long	.L7-.L4
	.long	.L29-.L4
	.long	.L6-.L4
	.long	.L5-.L4
	.long	.L3-.L4
	.text
.L13:
	movq	-56(%rbp), %rax
	jmp	.L18
.L8:
	movl	-64(%rbp), %eax
	movl	%eax, %edi
	call	createNode
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L19
.L3:
	movq	-56(%rbp), %rax
	jmp	.L18
.L10:
	cmpq	$0, -24(%rbp)
	je	.L20
	movq	$5, -16(%rbp)
	jmp	.L19
.L20:
	movq	$7, -16(%rbp)
	jmp	.L19
.L12:
	cmpl	$1, -60(%rbp)
	jne	.L22
	movq	$0, -16(%rbp)
	jmp	.L19
.L22:
	movq	$19, -16(%rbp)
	jmp	.L19
.L9:
	movq	-56(%rbp), %rax
	jmp	.L18
.L6:
	movq	-56(%rbp), %rax
	movq	%rax, -24(%rbp)
	movl	$1, -36(%rbp)
	movq	$17, -16(%rbp)
	jmp	.L19
.L7:
	movl	-60(%rbp), %eax
	subl	$1, %eax
	cmpl	%eax, -36(%rbp)
	jge	.L24
	movq	$11, -16(%rbp)
	jmp	.L19
.L24:
	movq	$7, -16(%rbp)
	jmp	.L19
.L15:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -24(%rbp)
	addl	$1, -36(%rbp)
	movq	$17, -16(%rbp)
	jmp	.L19
.L11:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rdx
	movq	-32(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	-24(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	$21, -16(%rbp)
	jmp	.L19
.L17:
	movq	-32(%rbp), %rax
	movq	-56(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-32(%rbp), %rax
	movq	%rax, -56(%rbp)
	movq	$13, -16(%rbp)
	jmp	.L19
.L14:
	cmpq	$0, -24(%rbp)
	jne	.L26
	movq	$2, -16(%rbp)
	jmp	.L19
.L26:
	movq	$10, -16(%rbp)
	jmp	.L19
.L16:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	-32(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$8, -16(%rbp)
	jmp	.L19
.L5:
	movq	$16, -16(%rbp)
	jmp	.L19
.L29:
	nop
.L19:
	jmp	.L28
.L18:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	insertAtPosition, .-insertAtPosition
	.globl	insertAtEnd
	.type	insertAtEnd, @function
insertAtEnd:
.LFB1:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movl	%esi, -44(%rbp)
	movq	$2, -16(%rbp)
.L48:
	cmpq	$10, -16(%rbp)
	ja	.L49
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L33(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L33(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L33:
	.long	.L41-.L33
	.long	.L40-.L33
	.long	.L39-.L33
	.long	.L38-.L33
	.long	.L37-.L33
	.long	.L36-.L33
	.long	.L35-.L33
	.long	.L49-.L33
	.long	.L49-.L33
	.long	.L34-.L33
	.long	.L32-.L33
	.text
.L37:
	movq	-24(%rbp), %rax
	movq	-32(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	$1, -16(%rbp)
	jmp	.L42
.L40:
	movq	-40(%rbp), %rax
	jmp	.L43
.L38:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	testq	%rax, %rax
	je	.L44
	movq	$6, -16(%rbp)
	jmp	.L42
.L44:
	movq	$4, -16(%rbp)
	jmp	.L42
.L34:
	movq	-40(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L42
.L35:
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -24(%rbp)
	movq	$3, -16(%rbp)
	jmp	.L42
.L36:
	movl	-44(%rbp), %eax
	movl	%eax, %edi
	call	createNode
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$10, -16(%rbp)
	jmp	.L42
.L32:
	cmpq	$0, -40(%rbp)
	jne	.L46
	movq	$0, -16(%rbp)
	jmp	.L42
.L46:
	movq	$9, -16(%rbp)
	jmp	.L42
.L41:
	movq	-32(%rbp), %rax
	jmp	.L43
.L39:
	movq	$5, -16(%rbp)
	jmp	.L42
.L49:
	nop
.L42:
	jmp	.L48
.L43:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE1:
	.size	insertAtEnd, .-insertAtEnd
	.globl	main
	.type	main, @function
main:
.LFB3:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	pushq	%rbx
	subq	$136, %rsp
	.cfi_offset 3, -24
	movl	%edi, -116(%rbp)
	movq	%rsi, -128(%rbp)
	movq	%rdx, -136(%rbp)
	movq	$0, _TIG_IZ_51Ly_envp(%rip)
	nop
.L51:
	movq	$0, _TIG_IZ_51Ly_argv(%rip)
	nop
.L52:
	movl	$0, _TIG_IZ_51Ly_argc(%rip)
	nop
	nop
.L53:
.L54:
#APP
# 131 "akmalscript_data-structure_coret.c" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-51Ly--0
# 0 "" 2
#NO_APP
	movl	-116(%rbp), %eax
	movl	%eax, _TIG_IZ_51Ly_argc(%rip)
	movq	-128(%rbp), %rax
	movq	%rax, _TIG_IZ_51Ly_argv(%rip)
	movq	-136(%rbp), %rax
	movq	%rax, _TIG_IZ_51Ly_envp(%rip)
	nop
	movq	$1, -88(%rbp)
.L60:
	cmpq	$2, -88(%rbp)
	je	.L55
	cmpq	$2, -88(%rbp)
	ja	.L62
	cmpq	$0, -88(%rbp)
	je	.L57
	cmpq	$1, -88(%rbp)
	jne	.L62
	movq	$2, -88(%rbp)
	jmp	.L58
.L57:
	movl	$0, %eax
	jmp	.L61
.L55:
	movl	$1, %edi
	call	createNode
	movq	%rax, -80(%rbp)
	movq	-80(%rbp), %rax
	movq	%rax, -72(%rbp)
	movl	$2, %edi
	call	createNode
	movq	%rax, -64(%rbp)
	movq	-64(%rbp), %rax
	movq	%rax, -56(%rbp)
	movl	$3, %edi
	call	createNode
	movq	%rax, -48(%rbp)
	movq	-48(%rbp), %rax
	movq	%rax, -40(%rbp)
	movq	-72(%rbp), %rax
	movq	-56(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	-56(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movl	$2, %edi
	call	createNode
	movq	%rax, -32(%rbp)
	movq	-32(%rbp), %rax
	movq	%rax, -24(%rbp)
	movl	$3, %edi
	call	createNode
	movq	-24(%rbp), %rdx
	movq	%rax, 8(%rdx)
	movq	-24(%rbp), %rax
	movq	8(%rax), %rbx
	movl	$5, %edi
	call	createNode
	movq	%rax, 8(%rbx)
	movq	-24(%rbp), %rax
	movq	8(%rax), %rax
	movq	8(%rax), %rbx
	movl	$6, %edi
	call	createNode
	movq	%rax, 8(%rbx)
	movl	$1, -108(%rbp)
	movl	-108(%rbp), %edx
	movq	-24(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	insertAtFront
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movl	$7, %esi
	movq	%rax, %rdi
	call	insertAtEnd
	movq	%rax, -24(%rbp)
	movl	$12, -104(%rbp)
	movl	$4, -100(%rbp)
	movl	-104(%rbp), %edx
	movl	-100(%rbp), %ecx
	movq	-24(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	insertAtPosition
	movq	%rax, -24(%rbp)
	movl	$12, -96(%rbp)
	movl	$4, -92(%rbp)
	movl	-92(%rbp), %edx
	movl	-96(%rbp), %ecx
	movq	-24(%rbp), %rax
	movl	%ecx, %esi
	movq	%rax, %rdi
	call	insertAfter
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, %rdi
	call	printList
	movq	-72(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-56(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	-40(%rbp), %rax
	movq	%rax, %rdi
	call	free@PLT
	movq	$0, -88(%rbp)
	jmp	.L58
.L62:
	nop
.L58:
	jmp	.L60
.L61:
	movq	-8(%rbp), %rbx
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE3:
	.size	main, .-main
	.section	.rodata
.LC1:
	.string	"%d "
	.text
	.globl	printList
	.type	printList, @function
printList:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movq	$2, -8(%rbp)
.L75:
	cmpq	$7, -8(%rbp)
	ja	.L76
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L66(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L66(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L66:
	.long	.L76-.L66
	.long	.L76-.L66
	.long	.L70-.L66
	.long	.L76-.L66
	.long	.L69-.L66
	.long	.L68-.L66
	.long	.L77-.L66
	.long	.L65-.L66
	.text
.L69:
	movq	-16(%rbp), %rax
	movl	(%rax), %eax
	movl	%eax, %esi
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	-16(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -16(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L71
.L68:
	movl	$10, %edi
	call	putchar@PLT
	movq	$6, -8(%rbp)
	jmp	.L71
.L65:
	cmpq	$0, -16(%rbp)
	je	.L73
	movq	$4, -8(%rbp)
	jmp	.L71
.L73:
	movq	$5, -8(%rbp)
	jmp	.L71
.L70:
	movq	-24(%rbp), %rax
	movq	%rax, -16(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L71
.L76:
	nop
.L71:
	jmp	.L75
.L77:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	printList, .-printList
	.globl	insertAtFront
	.type	insertAtFront, @function
insertAtFront:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movl	%esi, -44(%rbp)
	movq	$0, -16(%rbp)
.L84:
	cmpq	$2, -16(%rbp)
	je	.L79
	cmpq	$2, -16(%rbp)
	ja	.L86
	cmpq	$0, -16(%rbp)
	je	.L81
	cmpq	$1, -16(%rbp)
	jne	.L86
	movq	-24(%rbp), %rax
	jmp	.L85
.L81:
	movq	$2, -16(%rbp)
	jmp	.L83
.L79:
	movl	-44(%rbp), %eax
	movl	%eax, %edi
	call	createNode
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	-40(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	$1, -16(%rbp)
	jmp	.L83
.L86:
	nop
.L83:
	jmp	.L84
.L85:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	insertAtFront, .-insertAtFront
	.globl	insertAfter
	.type	insertAfter, @function
insertAfter:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movq	%rdi, -40(%rbp)
	movl	%esi, -44(%rbp)
	movl	%edx, -48(%rbp)
	movq	$4, -24(%rbp)
.L106:
	cmpq	$10, -24(%rbp)
	ja	.L107
	movq	-24(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L90(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L90(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L90:
	.long	.L97-.L90
	.long	.L96-.L90
	.long	.L95-.L90
	.long	.L94-.L90
	.long	.L93-.L90
	.long	.L107-.L90
	.long	.L92-.L90
	.long	.L107-.L90
	.long	.L107-.L90
	.long	.L91-.L90
	.long	.L89-.L90
	.text
.L93:
	movq	-40(%rbp), %rax
	movq	%rax, -32(%rbp)
	movq	$3, -24(%rbp)
	jmp	.L98
.L96:
	movq	-40(%rbp), %rax
	jmp	.L99
.L94:
	cmpq	$0, -32(%rbp)
	je	.L100
	movq	$10, -24(%rbp)
	jmp	.L98
.L100:
	movq	$9, -24(%rbp)
	jmp	.L98
.L91:
	cmpq	$0, -32(%rbp)
	jne	.L102
	movq	$6, -24(%rbp)
	jmp	.L98
.L102:
	movq	$0, -24(%rbp)
	jmp	.L98
.L92:
	movq	-40(%rbp), %rax
	jmp	.L99
.L89:
	movq	-32(%rbp), %rax
	movl	(%rax), %eax
	cmpl	%eax, -44(%rbp)
	jne	.L104
	movq	$9, -24(%rbp)
	jmp	.L98
.L104:
	movq	$2, -24(%rbp)
	jmp	.L98
.L97:
	movl	-48(%rbp), %eax
	movl	%eax, %edi
	call	createNode
	movq	%rax, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -8(%rbp)
	movq	-32(%rbp), %rax
	movq	8(%rax), %rdx
	movq	-8(%rbp), %rax
	movq	%rdx, 8(%rax)
	movq	-32(%rbp), %rax
	movq	-8(%rbp), %rdx
	movq	%rdx, 8(%rax)
	movq	$1, -24(%rbp)
	jmp	.L98
.L95:
	movq	-32(%rbp), %rax
	movq	8(%rax), %rax
	movq	%rax, -32(%rbp)
	movq	$3, -24(%rbp)
	jmp	.L98
.L107:
	nop
.L98:
	jmp	.L106
.L99:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	insertAfter, .-insertAfter
	.globl	createNode
	.type	createNode, @function
createNode:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$48, %rsp
	movl	%edi, -36(%rbp)
	movq	$1, -16(%rbp)
.L114:
	cmpq	$2, -16(%rbp)
	je	.L109
	cmpq	$2, -16(%rbp)
	ja	.L116
	cmpq	$0, -16(%rbp)
	je	.L111
	cmpq	$1, -16(%rbp)
	jne	.L116
	movq	$0, -16(%rbp)
	jmp	.L112
.L111:
	movl	$16, %edi
	call	malloc@PLT
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -24(%rbp)
	movq	-24(%rbp), %rax
	movl	-36(%rbp), %edx
	movl	%edx, (%rax)
	movq	-24(%rbp), %rax
	movq	$0, 8(%rax)
	movq	$2, -16(%rbp)
	jmp	.L112
.L109:
	movq	-24(%rbp), %rax
	jmp	.L115
.L116:
	nop
.L112:
	jmp	.L114
.L115:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	createNode, .-createNode
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
