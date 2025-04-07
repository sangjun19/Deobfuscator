	.file	"sankalp-happy_DSA_6_flatten.c"
	.text
	.globl	_TIG_IZ_5QNF_argv
	.bss
	.align 8
	.type	_TIG_IZ_5QNF_argv, @object
	.size	_TIG_IZ_5QNF_argv, 8
_TIG_IZ_5QNF_argv:
	.zero	8
	.globl	r
	.align 4
	.type	r, @object
	.size	r, 4
r:
	.zero	4
	.globl	count
	.align 4
	.type	count, @object
	.size	count, 4
count:
	.zero	4
	.globl	item
	.align 4
	.type	item, @object
	.size	item, 4
item:
	.zero	4
	.globl	_TIG_IZ_5QNF_envp
	.align 8
	.type	_TIG_IZ_5QNF_envp, @object
	.size	_TIG_IZ_5QNF_envp, 8
_TIG_IZ_5QNF_envp:
	.zero	8
	.globl	f
	.align 4
	.type	f, @object
	.size	f, 4
f:
	.zero	4
	.globl	q
	.align 16
	.type	q, @object
	.size	q, 16
q:
	.zero	16
	.globl	_TIG_IZ_5QNF_argc
	.align 4
	.type	_TIG_IZ_5QNF_argc, @object
	.size	_TIG_IZ_5QNF_argc, 4
_TIG_IZ_5QNF_argc:
	.zero	4
	.section	.rodata
.LC0:
	.string	"***********************"
.LC1:
	.string	"Circular Queue Operations"
.LC2:
	.string	"1. Insert"
.LC3:
	.string	"2. Delete"
.LC4:
	.string	"3. Display"
.LC5:
	.string	"4. Quit"
.LC6:
	.string	"Enter your choice:"
.LC7:
	.string	"%d"
.LC8:
	.string	"Invalid choice"
.LC9:
	.string	"Enter the item to be inserted"
	.text
	.globl	main
	.type	main, @function
main:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$64, %rsp
	movl	%edi, -36(%rbp)
	movq	%rsi, -48(%rbp)
	movq	%rdx, -56(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, item(%rip)
	nop
.L2:
	movl	$0, count(%rip)
	nop
.L3:
	movl	$0, f(%rip)
	nop
.L4:
	movl	$-1, r(%rip)
	nop
.L5:
	movl	$0, -20(%rbp)
	jmp	.L6
.L7:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	q(%rip), %rax
	movl	$0, (%rdx,%rax)
	addl	$1, -20(%rbp)
.L6:
	cmpl	$3, -20(%rbp)
	jle	.L7
	nop
.L8:
	movq	$0, _TIG_IZ_5QNF_envp(%rip)
	nop
.L9:
	movq	$0, _TIG_IZ_5QNF_argv(%rip)
	nop
.L10:
	movl	$0, _TIG_IZ_5QNF_argc(%rip)
	nop
	nop
.L11:
.L12:
#APP
# 143 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-5QNF--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_5QNF_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_5QNF_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_5QNF_envp(%rip)
	nop
	movq	$1, -16(%rbp)
.L35:
	cmpq	$13, -16(%rbp)
	ja	.L38
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L15(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L15(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L15:
	.long	.L38-.L15
	.long	.L24-.L15
	.long	.L23-.L15
	.long	.L22-.L15
	.long	.L38-.L15
	.long	.L21-.L15
	.long	.L20-.L15
	.long	.L19-.L15
	.long	.L18-.L15
	.long	.L17-.L15
	.long	.L16-.L15
	.long	.L38-.L15
	.long	.L38-.L15
	.long	.L14-.L15
	.text
.L18:
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$9, -16(%rbp)
	jmp	.L25
.L24:
	movq	$8, -16(%rbp)
	jmp	.L25
.L22:
	movl	f(%rip), %eax
	movl	%eax, %edi
	call	display
	movq	$6, -16(%rbp)
	jmp	.L25
.L17:
	movl	-24(%rbp), %eax
	cmpl	$4, %eax
	je	.L26
	cmpl	$4, %eax
	jg	.L27
	cmpl	$3, %eax
	je	.L28
	cmpl	$3, %eax
	jg	.L27
	cmpl	$1, %eax
	je	.L29
	cmpl	$2, %eax
	je	.L30
	jmp	.L27
.L26:
	movq	$2, -16(%rbp)
	jmp	.L31
.L28:
	movq	$3, -16(%rbp)
	jmp	.L31
.L30:
	movq	$5, -16(%rbp)
	jmp	.L31
.L29:
	movq	$7, -16(%rbp)
	jmp	.L31
.L27:
	movq	$10, -16(%rbp)
	nop
.L31:
	jmp	.L25
.L14:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L36
	jmp	.L37
.L20:
	movl	-24(%rbp), %eax
	cmpl	$4, %eax
	je	.L33
	movq	$8, -16(%rbp)
	jmp	.L25
.L33:
	movq	$13, -16(%rbp)
	jmp	.L25
.L21:
	call	del
	movq	$6, -16(%rbp)
	jmp	.L25
.L16:
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$6, -16(%rbp)
	jmp	.L25
.L19:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	leaq	item(%rip), %rax
	movq	%rax, %rsi
	leaq	.LC7(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	call	insert
	movq	$6, -16(%rbp)
	jmp	.L25
.L23:
	movl	$0, %edi
	call	exit@PLT
.L38:
	nop
.L25:
	jmp	.L35
.L37:
	call	__stack_chk_fail@PLT
.L36:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	main, .-main
	.section	.rodata
.LC10:
	.string	"Queue UnderFlow"
.LC11:
	.string	"The Deleted item is %d\n"
	.text
	.globl	del
	.type	del, @function
del:
.LFB4:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$5, -8(%rbp)
.L51:
	cmpq	$5, -8(%rbp)
	ja	.L52
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L42(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L42(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L42:
	.long	.L52-.L42
	.long	.L46-.L42
	.long	.L53-.L42
	.long	.L53-.L42
	.long	.L43-.L42
	.long	.L41-.L42
	.text
.L43:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$2, -8(%rbp)
	jmp	.L47
.L46:
	movl	f(%rip), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	q(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	f(%rip), %eax
	addl	$1, %eax
	cltd
	shrl	$30, %edx
	addl	%edx, %eax
	andl	$3, %eax
	subl	%edx, %eax
	movl	%eax, f(%rip)
	movl	count(%rip), %eax
	subl	$1, %eax
	movl	%eax, count(%rip)
	movq	$3, -8(%rbp)
	jmp	.L47
.L41:
	movl	count(%rip), %eax
	testl	%eax, %eax
	jne	.L49
	movq	$4, -8(%rbp)
	jmp	.L47
.L49:
	movq	$1, -8(%rbp)
	jmp	.L47
.L52:
	nop
.L47:
	jmp	.L51
.L53:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	del, .-del
	.section	.rodata
.LC12:
	.string	"Queue OverFlow"
	.text
	.globl	insert
	.type	insert, @function
insert:
.LFB6:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$2, -8(%rbp)
.L66:
	cmpq	$5, -8(%rbp)
	ja	.L67
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L57(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L57(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L57:
	.long	.L61-.L57
	.long	.L68-.L57
	.long	.L59-.L57
	.long	.L67-.L57
	.long	.L68-.L57
	.long	.L56-.L57
	.text
.L56:
	movl	r(%rip), %eax
	addl	$1, %eax
	cltd
	shrl	$30, %edx
	addl	%edx, %eax
	andl	$3, %eax
	subl	%edx, %eax
	movl	%eax, r(%rip)
	movl	r(%rip), %edx
	movl	item(%rip), %eax
	movslq	%edx, %rdx
	leaq	0(,%rdx,4), %rcx
	leaq	q(%rip), %rdx
	movl	%eax, (%rcx,%rdx)
	movl	count(%rip), %eax
	addl	$1, %eax
	movl	%eax, count(%rip)
	movq	$1, -8(%rbp)
	jmp	.L63
.L61:
	leaq	.LC12(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$4, -8(%rbp)
	jmp	.L63
.L59:
	movl	count(%rip), %eax
	cmpl	$4, %eax
	jne	.L64
	movq	$0, -8(%rbp)
	jmp	.L63
.L64:
	movq	$5, -8(%rbp)
	jmp	.L63
.L67:
	nop
.L63:
	jmp	.L66
.L68:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	insert, .-insert
	.section	.rodata
.LC13:
	.string	"Contents of the queue"
.LC14:
	.string	"Queue is Empty"
.LC15:
	.string	"%d\n"
	.text
	.globl	display
	.type	display, @function
display:
.LFB8:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movl	%edi, -20(%rbp)
	movq	$1, -8(%rbp)
.L85:
	cmpq	$9, -8(%rbp)
	ja	.L86
	movq	-8(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L72(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L72(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L72:
	.long	.L78-.L72
	.long	.L77-.L72
	.long	.L76-.L72
	.long	.L86-.L72
	.long	.L86-.L72
	.long	.L75-.L72
	.long	.L87-.L72
	.long	.L73-.L72
	.long	.L86-.L72
	.long	.L87-.L72
	.text
.L77:
	movl	count(%rip), %eax
	testl	%eax, %eax
	jne	.L79
	movq	$0, -8(%rbp)
	jmp	.L81
.L79:
	movq	$5, -8(%rbp)
	jmp	.L81
.L75:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	$1, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L81
.L78:
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$6, -8(%rbp)
	jmp	.L81
.L73:
	movl	count(%rip), %eax
	cmpl	%eax, -12(%rbp)
	jg	.L83
	movq	$2, -8(%rbp)
	jmp	.L81
.L83:
	movq	$9, -8(%rbp)
	jmp	.L81
.L76:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	leaq	q(%rip), %rax
	movl	(%rdx,%rax), %eax
	movl	%eax, %esi
	leaq	.LC15(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-20(%rbp), %eax
	addl	$1, %eax
	cltd
	shrl	$30, %edx
	addl	%edx, %eax
	andl	$3, %eax
	subl	%edx, %eax
	movl	%eax, -20(%rbp)
	addl	$1, -12(%rbp)
	movq	$7, -8(%rbp)
	jmp	.L81
.L86:
	nop
.L81:
	jmp	.L85
.L87:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE8:
	.size	display, .-display
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
