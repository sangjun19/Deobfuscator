	.file	"vachanmn123_oslab_3_flatten.c"
	.text
	.globl	_TIG_IZ_ZE93_envp
	.bss
	.align 8
	.type	_TIG_IZ_ZE93_envp, @object
	.size	_TIG_IZ_ZE93_envp, 8
_TIG_IZ_ZE93_envp:
	.zero	8
	.globl	_TIG_IZ_ZE93_argc
	.align 4
	.type	_TIG_IZ_ZE93_argc, @object
	.size	_TIG_IZ_ZE93_argc, 4
_TIG_IZ_ZE93_argc:
	.zero	4
	.globl	_TIG_IZ_ZE93_argv
	.align 8
	.type	_TIG_IZ_ZE93_argv, @object
	.size	_TIG_IZ_ZE93_argv, 8
_TIG_IZ_ZE93_argv:
	.zero	8
	.globl	full
	.align 4
	.type	full, @object
	.size	full, 4
full:
	.zero	4
	.globl	empty
	.align 4
	.type	empty, @object
	.size	empty, 4
empty:
	.zero	4
	.globl	mutex
	.align 4
	.type	mutex, @object
	.size	mutex, 4
mutex:
	.zero	4
	.globl	x
	.align 4
	.type	x, @object
	.size	x, 4
x:
	.zero	4
	.section	.rodata
.LC0:
	.string	"\nproducer produces item %d"
	.text
	.globl	producer
	.type	producer, @function
producer:
.LFB2:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$0, -8(%rbp)
.L7:
	cmpq	$2, -8(%rbp)
	je	.L8
	cmpq	$2, -8(%rbp)
	ja	.L9
	cmpq	$0, -8(%rbp)
	je	.L4
	cmpq	$1, -8(%rbp)
	jne	.L9
	movl	mutex(%rip), %eax
	subl	$1, %eax
	movl	%eax, mutex(%rip)
	movl	full(%rip), %eax
	addl	$1, %eax
	movl	%eax, full(%rip)
	movl	empty(%rip), %eax
	subl	$1, %eax
	movl	%eax, empty(%rip)
	movl	x(%rip), %eax
	addl	$1, %eax
	movl	%eax, x(%rip)
	movl	x(%rip), %eax
	movl	%eax, %esi
	leaq	.LC0(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	mutex(%rip), %eax
	addl	$1, %eax
	movl	%eax, mutex(%rip)
	movq	$2, -8(%rbp)
	jmp	.L5
.L4:
	movq	$1, -8(%rbp)
	jmp	.L5
.L9:
	nop
.L5:
	jmp	.L7
.L8:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE2:
	.size	producer, .-producer
	.section	.rodata
.LC1:
	.string	"\n enter your choice:"
.LC2:
	.string	"%d"
.LC3:
	.string	"buffer is full!"
	.align 8
.LC4:
	.string	"\n1.press 1 for producer\n2.press 2 for consumer\n3.press 3 for exit"
.LC5:
	.string	"buffer is empty!"
	.text
	.globl	main
	.type	main, @function
main:
.LFB4:
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
	movl	$0, x(%rip)
	nop
.L11:
	movl	$3, empty(%rip)
	nop
.L12:
	movl	$0, full(%rip)
	nop
.L13:
	movl	$1, mutex(%rip)
	nop
.L14:
	movq	$0, _TIG_IZ_ZE93_envp(%rip)
	nop
.L15:
	movq	$0, _TIG_IZ_ZE93_argv(%rip)
	nop
.L16:
	movl	$0, _TIG_IZ_ZE93_argc(%rip)
	nop
	nop
.L17:
.L18:
#APP
# 114 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-ZE93--0
# 0 "" 2
#NO_APP
	movl	-36(%rbp), %eax
	movl	%eax, _TIG_IZ_ZE93_argc(%rip)
	movq	-48(%rbp), %rax
	movq	%rax, _TIG_IZ_ZE93_argv(%rip)
	movq	-56(%rbp), %rax
	movq	%rax, _TIG_IZ_ZE93_envp(%rip)
	nop
	movq	$5, -16(%rbp)
.L57:
	cmpq	$24, -16(%rbp)
	ja	.L60
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L21(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L21(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L21:
	.long	.L60-.L21
	.long	.L39-.L21
	.long	.L38-.L21
	.long	.L60-.L21
	.long	.L37-.L21
	.long	.L36-.L21
	.long	.L60-.L21
	.long	.L35-.L21
	.long	.L34-.L21
	.long	.L33-.L21
	.long	.L32-.L21
	.long	.L31-.L21
	.long	.L30-.L21
	.long	.L29-.L21
	.long	.L28-.L21
	.long	.L27-.L21
	.long	.L60-.L21
	.long	.L60-.L21
	.long	.L26-.L21
	.long	.L25-.L21
	.long	.L24-.L21
	.long	.L23-.L21
	.long	.L22-.L21
	.long	.L60-.L21
	.long	.L20-.L21
	.text
.L26:
	movl	-24(%rbp), %eax
	cmpl	$3, %eax
	je	.L40
	cmpl	$3, %eax
	jg	.L41
	cmpl	$1, %eax
	je	.L42
	cmpl	$2, %eax
	je	.L43
	jmp	.L41
.L40:
	movq	$2, -16(%rbp)
	jmp	.L44
.L43:
	movq	$20, -16(%rbp)
	jmp	.L44
.L42:
	movq	$1, -16(%rbp)
	jmp	.L44
.L41:
	movq	$13, -16(%rbp)
	nop
.L44:
	jmp	.L45
.L37:
	movl	empty(%rip), %eax
	testl	%eax, %eax
	je	.L46
	movq	$10, -16(%rbp)
	jmp	.L45
.L46:
	movq	$22, -16(%rbp)
	jmp	.L45
.L28:
	leaq	.LC1(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-24(%rbp), %rax
	movq	%rax, %rsi
	leaq	.LC2(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	__isoc99_scanf@PLT
	movq	$18, -16(%rbp)
	jmp	.L45
.L27:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -16(%rbp)
	jmp	.L45
.L30:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	$1, -20(%rbp)
	movq	$21, -16(%rbp)
	jmp	.L45
.L34:
	addl	$1, -20(%rbp)
	movq	$21, -16(%rbp)
	jmp	.L45
.L39:
	movl	mutex(%rip), %eax
	cmpl	$1, %eax
	jne	.L48
	movq	$4, -16(%rbp)
	jmp	.L45
.L48:
	movq	$15, -16(%rbp)
	jmp	.L45
.L20:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -16(%rbp)
	jmp	.L45
.L23:
	cmpl	$0, -20(%rbp)
	jle	.L50
	movq	$14, -16(%rbp)
	jmp	.L45
.L50:
	movq	$7, -16(%rbp)
	jmp	.L45
.L31:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -16(%rbp)
	jmp	.L45
.L33:
	movl	full(%rip), %eax
	testl	%eax, %eax
	je	.L52
	movq	$19, -16(%rbp)
	jmp	.L45
.L52:
	movq	$11, -16(%rbp)
	jmp	.L45
.L29:
	movq	$8, -16(%rbp)
	jmp	.L45
.L25:
	call	consumer
	movq	$8, -16(%rbp)
	jmp	.L45
.L22:
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$8, -16(%rbp)
	jmp	.L45
.L36:
	movq	$12, -16(%rbp)
	jmp	.L45
.L32:
	call	producer
	movq	$8, -16(%rbp)
	jmp	.L45
.L35:
	movl	$0, %eax
	movq	-8(%rbp), %rdx
	subq	%fs:40, %rdx
	je	.L58
	jmp	.L59
.L38:
	movl	$0, %edi
	call	exit@PLT
.L24:
	movl	mutex(%rip), %eax
	cmpl	$1, %eax
	jne	.L55
	movq	$9, -16(%rbp)
	jmp	.L45
.L55:
	movq	$24, -16(%rbp)
	jmp	.L45
.L60:
	nop
.L45:
	jmp	.L57
.L59:
	call	__stack_chk_fail@PLT
.L58:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE4:
	.size	main, .-main
	.section	.rodata
.LC6:
	.string	"\nconsumer consumes item %d"
	.text
	.globl	consumer
	.type	consumer, @function
consumer:
.LFB7:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$16, %rsp
	movq	$2, -8(%rbp)
.L67:
	cmpq	$2, -8(%rbp)
	je	.L62
	cmpq	$2, -8(%rbp)
	ja	.L68
	cmpq	$0, -8(%rbp)
	je	.L69
	cmpq	$1, -8(%rbp)
	jne	.L68
	movl	mutex(%rip), %eax
	subl	$1, %eax
	movl	%eax, mutex(%rip)
	movl	full(%rip), %eax
	subl	$1, %eax
	movl	%eax, full(%rip)
	movl	empty(%rip), %eax
	addl	$1, %eax
	movl	%eax, empty(%rip)
	movl	x(%rip), %eax
	movl	%eax, %esi
	leaq	.LC6(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	x(%rip), %eax
	subl	$1, %eax
	movl	%eax, x(%rip)
	movl	mutex(%rip), %eax
	addl	$1, %eax
	movl	%eax, mutex(%rip)
	movq	$0, -8(%rbp)
	jmp	.L65
.L62:
	movq	$1, -8(%rbp)
	jmp	.L65
.L68:
	nop
.L65:
	jmp	.L67
.L69:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE7:
	.size	consumer, .-consumer
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
