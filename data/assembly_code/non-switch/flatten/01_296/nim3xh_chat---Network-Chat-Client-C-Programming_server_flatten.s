	.file	"nim3xh_chat---Network-Chat-Client-C-Programming_server_flatten.c"
	.text
	.globl	_TIG_IZ_GCfI_argc
	.bss
	.align 4
	.type	_TIG_IZ_GCfI_argc, @object
	.size	_TIG_IZ_GCfI_argc, 4
_TIG_IZ_GCfI_argc:
	.zero	4
	.globl	userCounter
	.align 4
	.type	userCounter, @object
	.size	userCounter, 4
userCounter:
	.zero	4
	.globl	_TIG_IZ_GCfI_envp
	.align 8
	.type	_TIG_IZ_GCfI_envp, @object
	.size	_TIG_IZ_GCfI_envp, 8
_TIG_IZ_GCfI_envp:
	.zero	8
	.globl	_TIG_IZ_GCfI_argv
	.align 8
	.type	_TIG_IZ_GCfI_argv, @object
	.size	_TIG_IZ_GCfI_argv, 8
_TIG_IZ_GCfI_argv:
	.zero	8
	.section	.rodata
.LC0:
	.string	"User%d"
	.text
	.globl	generateDefaultName
	.type	generateDefaultName, @function
generateDefaultName:
.LFB0:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$32, %rsp
	movq	%rdi, -24(%rbp)
	movl	%esi, -28(%rbp)
	movq	$2, -8(%rbp)
.L7:
	cmpq	$2, -8(%rbp)
	je	.L2
	cmpq	$2, -8(%rbp)
	ja	.L9
	cmpq	$0, -8(%rbp)
	je	.L4
	cmpq	$1, -8(%rbp)
	jne	.L9
	jmp	.L8
.L4:
	movl	-28(%rbp), %edx
	movq	-24(%rbp), %rax
	movl	%edx, %ecx
	leaq	.LC0(%rip), %rdx
	movl	$64, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	movl	userCounter(%rip), %eax
	addl	$1, %eax
	movl	%eax, userCounter(%rip)
	movq	$1, -8(%rbp)
	jmp	.L6
.L2:
	movq	$0, -8(%rbp)
	jmp	.L6
.L9:
	nop
.L6:
	jmp	.L7
.L8:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE0:
	.size	generateDefaultName, .-generateDefaultName
	.section	.rodata
	.align 8
.LC1:
	.string	"%s has changed their name to %s"
	.text
	.globl	broadcastNameChange
	.type	broadcastNameChange, @function
broadcastNameChange:
.LFB5:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$208, %rsp
	movl	%edi, -164(%rbp)
	movq	%rsi, -176(%rbp)
	movq	%rdx, -184(%rbp)
	movq	%rcx, -192(%rbp)
	movq	%r8, -200(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$0, -152(%rbp)
.L16:
	cmpq	$2, -152(%rbp)
	je	.L19
	cmpq	$2, -152(%rbp)
	ja	.L20
	cmpq	$0, -152(%rbp)
	je	.L13
	cmpq	$1, -152(%rbp)
	jne	.L20
	movq	-200(%rbp), %rcx
	movq	-192(%rbp), %rdx
	leaq	-144(%rbp), %rax
	movq	%rcx, %r8
	movq	%rdx, %rcx
	leaq	.LC1(%rip), %rdx
	movl	$128, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-144(%rbp), %rcx
	movq	-184(%rbp), %rdx
	movq	-176(%rbp), %rsi
	movl	-164(%rbp), %eax
	movl	%eax, %edi
	call	broadcast
	movq	$2, -152(%rbp)
	jmp	.L14
.L13:
	movq	$1, -152(%rbp)
	jmp	.L14
.L20:
	nop
.L14:
	jmp	.L16
.L19:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L18
	call	__stack_chk_fail@PLT
.L18:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE5:
	.size	broadcastNameChange, .-broadcastNameChange
	.globl	broadcast
	.type	broadcast, @function
broadcast:
.LFB6:
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
	movq	%rcx, -64(%rbp)
	movq	$0, -16(%rbp)
.L42:
	cmpq	$10, -16(%rbp)
	ja	.L43
	movq	-16(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L24(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L24(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L24:
	.long	.L31-.L24
	.long	.L30-.L24
	.long	.L43-.L24
	.long	.L29-.L24
	.long	.L43-.L24
	.long	.L44-.L24
	.long	.L27-.L24
	.long	.L43-.L24
	.long	.L26-.L24
	.long	.L25-.L24
	.long	.L23-.L24
	.text
.L26:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	%eax, -36(%rbp)
	je	.L32
	movq	$1, -16(%rbp)
	jmp	.L34
.L32:
	movq	$10, -16(%rbp)
	jmp	.L34
.L30:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	leal	63(%rax), %edx
	testl	%eax, %eax
	cmovs	%edx, %eax
	sarl	$6, %eax
	movl	%eax, %edx
	movq	-56(%rbp), %rax
	movslq	%edx, %rdx
	movq	(%rax,%rdx,8), %rdx
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rcx
	movq	-48(%rbp), %rax
	addq	%rcx, %rax
	movl	(%rax), %eax
	andl	$63, %eax
	movl	$1, %esi
	movl	%eax, %ecx
	salq	%cl, %rsi
	movq	%rsi, %rax
	andq	%rdx, %rax
	testq	%rax, %rax
	je	.L35
	movq	$3, -16(%rbp)
	jmp	.L34
.L35:
	movq	$10, -16(%rbp)
	jmp	.L34
.L29:
	movq	-64(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -8(%rbp)
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	movq	-8(%rbp), %rdx
	movq	-64(%rbp), %rsi
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	$10, -16(%rbp)
	jmp	.L34
.L25:
	cmpl	$9, -20(%rbp)
	jg	.L37
	movq	$6, -16(%rbp)
	jmp	.L34
.L37:
	movq	$5, -16(%rbp)
	jmp	.L34
.L27:
	movl	-20(%rbp), %eax
	cltq
	leaq	0(,%rax,4), %rdx
	movq	-48(%rbp), %rax
	addq	%rdx, %rax
	movl	(%rax), %eax
	cmpl	$-1, %eax
	je	.L39
	movq	$8, -16(%rbp)
	jmp	.L34
.L39:
	movq	$10, -16(%rbp)
	jmp	.L34
.L23:
	addl	$1, -20(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L34
.L31:
	movl	$0, -20(%rbp)
	movq	$9, -16(%rbp)
	jmp	.L34
.L43:
	nop
.L34:
	jmp	.L42
.L44:
	nop
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE6:
	.size	broadcast, .-broadcast
	.section	.rodata
.LC2:
	.string	"name"
.LC3:
	.string	"User %d disconnected.\n"
.LC4:
	.string	"Socket creation failed"
.LC5:
	.string	"Listen failed"
.LC6:
	.string	" "
.LC7:
	.string	"%s has quit"
.LC8:
	.string	"Client %d disconnected.\n"
	.align 8
.LC9:
	.string	"Invalid name format. Usage: name <new_name>"
.LC10:
	.string	"Receive error"
.LC11:
	.string	"Received from client %d: %s\n"
.LC12:
	.string	"quit"
.LC13:
	.string	"Select failed"
.LC14:
	.string	"New client connected"
.LC15:
	.string	"%s has connected"
.LC16:
	.string	"Usage: %s <port>\n"
	.align 8
.LC17:
	.string	"Server is listening on port %s...\n"
.LC18:
	.string	"Bind failed"
	.text
	.globl	main
	.type	main, @function
main:
.LFB9:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$1920, %rsp
	movl	%edi, -1892(%rbp)
	movq	%rsi, -1904(%rbp)
	movq	%rdx, -1912(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movl	$0, userCounter(%rip)
	nop
.L46:
	movq	$0, _TIG_IZ_GCfI_envp(%rip)
	nop
.L47:
	movq	$0, _TIG_IZ_GCfI_argv(%rip)
	nop
.L48:
	movl	$0, _TIG_IZ_GCfI_argc(%rip)
	nop
	nop
.L49:
.L50:
#APP
# 142 "/usr/include/x86_64-linux-gnu/bits/byteswap.h" 1
	##_ANNOTATION_INITIALREGION_-TIG-IZ-GCfI--0
# 0 "" 2
#NO_APP
	movl	-1892(%rbp), %eax
	movl	%eax, _TIG_IZ_GCfI_argc(%rip)
	movq	-1904(%rbp), %rax
	movq	%rax, _TIG_IZ_GCfI_argv(%rip)
	movq	-1912(%rbp), %rax
	movq	%rax, _TIG_IZ_GCfI_envp(%rip)
	nop
	movq	$85, -1800(%rbp)
.L154:
	cmpq	$95, -1800(%rbp)
	ja	.L156
	movq	-1800(%rbp), %rax
	leaq	0(,%rax,4), %rdx
	leaq	.L53(%rip), %rax
	movl	(%rdx,%rax), %eax
	cltq
	leaq	.L53(%rip), %rdx
	addq	%rdx, %rax
	notrack jmp	*%rax
	.section	.rodata
	.align 4
	.align 4
.L53:
	.long	.L156-.L53
	.long	.L108-.L53
	.long	.L156-.L53
	.long	.L156-.L53
	.long	.L156-.L53
	.long	.L156-.L53
	.long	.L107-.L53
	.long	.L106-.L53
	.long	.L105-.L53
	.long	.L104-.L53
	.long	.L156-.L53
	.long	.L103-.L53
	.long	.L102-.L53
	.long	.L156-.L53
	.long	.L101-.L53
	.long	.L156-.L53
	.long	.L156-.L53
	.long	.L100-.L53
	.long	.L99-.L53
	.long	.L98-.L53
	.long	.L156-.L53
	.long	.L97-.L53
	.long	.L156-.L53
	.long	.L96-.L53
	.long	.L95-.L53
	.long	.L156-.L53
	.long	.L94-.L53
	.long	.L156-.L53
	.long	.L93-.L53
	.long	.L156-.L53
	.long	.L156-.L53
	.long	.L156-.L53
	.long	.L92-.L53
	.long	.L91-.L53
	.long	.L156-.L53
	.long	.L90-.L53
	.long	.L89-.L53
	.long	.L88-.L53
	.long	.L87-.L53
	.long	.L86-.L53
	.long	.L85-.L53
	.long	.L84-.L53
	.long	.L83-.L53
	.long	.L82-.L53
	.long	.L156-.L53
	.long	.L81-.L53
	.long	.L80-.L53
	.long	.L79-.L53
	.long	.L78-.L53
	.long	.L77-.L53
	.long	.L156-.L53
	.long	.L156-.L53
	.long	.L156-.L53
	.long	.L76-.L53
	.long	.L156-.L53
	.long	.L156-.L53
	.long	.L75-.L53
	.long	.L74-.L53
	.long	.L73-.L53
	.long	.L72-.L53
	.long	.L156-.L53
	.long	.L71-.L53
	.long	.L70-.L53
	.long	.L156-.L53
	.long	.L69-.L53
	.long	.L156-.L53
	.long	.L156-.L53
	.long	.L156-.L53
	.long	.L68-.L53
	.long	.L67-.L53
	.long	.L66-.L53
	.long	.L156-.L53
	.long	.L156-.L53
	.long	.L65-.L53
	.long	.L64-.L53
	.long	.L63-.L53
	.long	.L156-.L53
	.long	.L62-.L53
	.long	.L156-.L53
	.long	.L156-.L53
	.long	.L61-.L53
	.long	.L156-.L53
	.long	.L60-.L53
	.long	.L59-.L53
	.long	.L156-.L53
	.long	.L58-.L53
	.long	.L156-.L53
	.long	.L156-.L53
	.long	.L156-.L53
	.long	.L156-.L53
	.long	.L57-.L53
	.long	.L56-.L53
	.long	.L55-.L53
	.long	.L156-.L53
	.long	.L54-.L53
	.long	.L52-.L53
	.text
.L99:
	cmpq	$0, -1808(%rbp)
	je	.L109
	movq	$56, -1800(%rbp)
	jmp	.L111
.L109:
	movq	$11, -1800(%rbp)
	jmp	.L111
.L61:
	cmpl	$-1, -1876(%rbp)
	jne	.L112
	movq	$57, -1800(%rbp)
	jmp	.L111
.L112:
	movq	$39, -1800(%rbp)
	jmp	.L111
.L77:
	addl	$1, -1836(%rbp)
	movq	$6, -1800(%rbp)
	jmp	.L111
.L70:
	movl	-1860(%rbp), %eax
	cmpl	-1884(%rbp), %eax
	jne	.L114
	movq	$41, -1800(%rbp)
	jmp	.L111
.L114:
	movq	$19, -1800(%rbp)
	jmp	.L111
.L101:
	leaq	-1040(%rbp), %rax
	movl	$4, %edx
	leaq	.LC2(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -1844(%rbp)
	movq	$82, -1800(%rbp)
	jmp	.L111
.L60:
	cmpl	$0, -1844(%rbp)
	jne	.L116
	movq	$74, -1800(%rbp)
	jmp	.L111
.L116:
	movq	$91, -1800(%rbp)
	jmp	.L111
.L75:
	movl	-1860(%rbp), %edx
	leaq	-1360(%rbp), %rax
	movl	%edx, %ecx
	leaq	.LC0(%rip), %rdx
	movl	$64, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	movq	-1808(%rbp), %rdi
	leaq	-1360(%rbp), %rcx
	leaq	-1680(%rbp), %rdx
	leaq	-1728(%rbp), %rsi
	movl	-1860(%rbp), %eax
	movq	%rdi, %r8
	movl	%eax, %edi
	call	broadcastNameChange
	movq	$69, -1800(%rbp)
	jmp	.L111
.L102:
	movl	-1884(%rbp), %eax
	movl	$10, %esi
	movl	%eax, %edi
	call	listen@PLT
	movl	%eax, -1876(%rbp)
	movq	$80, -1800(%rbp)
	jmp	.L111
.L67:
	addl	$1, -1860(%rbp)
	movq	$36, -1800(%rbp)
	jmp	.L111
.L105:
	movl	-1848(%rbp), %eax
	cltq
	movl	$-1, -1728(%rbp,%rax,4)
	movq	$69, -1800(%rbp)
	jmp	.L111
.L81:
	movl	userCounter(%rip), %eax
	movl	%eax, %esi
	leaq	.LC3(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-1860(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	-1860(%rbp), %eax
	leal	63(%rax), %edx
	testl	%eax, %eax
	cmovs	%edx, %eax
	sarl	$6, %eax
	movl	%eax, %esi
	movslq	%esi, %rax
	movq	-1680(%rbp,%rax,8), %rdx
	movl	-1860(%rbp), %eax
	andl	$63, %eax
	movl	$1, %edi
	movl	%eax, %ecx
	salq	%cl, %rdi
	movq	%rdi, %rax
	notq	%rax
	andq	%rax, %rdx
	movslq	%esi, %rax
	movq	%rdx, -1680(%rbp,%rax,8)
	movl	$0, -1836(%rbp)
	movq	$6, -1800(%rbp)
	jmp	.L111
.L108:
	movl	-1860(%rbp), %eax
	leal	63(%rax), %edx
	testl	%eax, %eax
	cmovs	%edx, %eax
	sarl	$6, %eax
	cltq
	movq	-1552(%rbp,%rax,8), %rdx
	movl	-1860(%rbp), %eax
	andl	$63, %eax
	movl	$1, %esi
	movl	%eax, %ecx
	salq	%cl, %rsi
	movq	%rsi, %rax
	andq	%rdx, %rax
	testq	%rax, %rax
	je	.L118
	movq	$62, -1800(%rbp)
	jmp	.L111
.L118:
	movq	$69, -1800(%rbp)
	jmp	.L111
.L96:
	cmpl	$0, -1840(%rbp)
	jne	.L120
	movq	$26, -1800(%rbp)
	jmp	.L111
.L120:
	movq	$14, -1800(%rbp)
	jmp	.L111
.L62:
	leaq	.LC4(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	$1, %edi
	call	exit@PLT
.L66:
	leaq	-1760(%rbp), %rax
	movl	$16, %edx
	movl	$0, %esi
	movq	%rax, %rdi
	call	memset@PLT
	movw	$2, -1760(%rbp)
	movl	$0, -1756(%rbp)
	movq	-1904(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rdi
	call	atoi@PLT
	movl	%eax, -1832(%rbp)
	movl	-1832(%rbp), %eax
	movzwl	%ax, %eax
	movl	%eax, %edi
	call	htons@PLT
	movw	%ax, -1758(%rbp)
	leaq	-1760(%rbp), %rcx
	movl	-1884(%rbp), %eax
	movl	$16, %edx
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	bind@PLT
	movl	%eax, -1880(%rbp)
	movq	$46, -1800(%rbp)
	jmp	.L111
.L95:
	cmpq	$0, -1816(%rbp)
	jle	.L122
	movq	$17, -1800(%rbp)
	jmp	.L111
.L122:
	movq	$32, -1800(%rbp)
	jmp	.L111
.L97:
	leaq	-1680(%rbp), %rax
	movq	%rax, -1824(%rbp)
	movl	$0, -1872(%rbp)
	movq	$92, -1800(%rbp)
	jmp	.L111
.L54:
	movl	-1868(%rbp), %eax
	cltq
	movl	$-1, -1728(%rbp,%rax,4)
	addl	$1, -1868(%rbp)
	movq	$68, -1800(%rbp)
	jmp	.L111
.L89:
	cmpl	$1023, -1860(%rbp)
	jg	.L124
	movq	$1, -1800(%rbp)
	jmp	.L111
.L124:
	movq	$58, -1800(%rbp)
	jmp	.L111
.L74:
	leaq	.LC5(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	-1884(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	$1, %edi
	call	exit@PLT
.L68:
	cmpl	$9, -1868(%rbp)
	jg	.L126
	movq	$94, -1800(%rbp)
	jmp	.L111
.L126:
	movq	$58, -1800(%rbp)
	jmp	.L111
.L58:
	cmpl	$2, -1892(%rbp)
	je	.L128
	movq	$95, -1800(%rbp)
	jmp	.L111
.L128:
	movq	$7, -1800(%rbp)
	jmp	.L111
.L94:
	leaq	-1040(%rbp), %rax
	addq	$5, %rax
	leaq	.LC6(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strtok@PLT
	movq	%rax, -1784(%rbp)
	movq	-1784(%rbp), %rax
	movq	%rax, -1776(%rbp)
	movq	-1776(%rbp), %rdx
	leaq	-1168(%rbp), %rax
	movq	%rdx, %rcx
	leaq	.LC7(%rip), %rdx
	movl	$128, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-1168(%rbp), %rcx
	leaq	-1680(%rbp), %rdx
	leaq	-1728(%rbp), %rsi
	movl	-1860(%rbp), %eax
	movl	%eax, %edi
	call	broadcast
	movl	-1860(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC8(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movl	-1860(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	-1860(%rbp), %eax
	leal	63(%rax), %edx
	testl	%eax, %eax
	cmovs	%edx, %eax
	sarl	$6, %eax
	movl	%eax, %esi
	movslq	%esi, %rax
	movq	-1680(%rbp,%rax,8), %rdx
	movl	-1860(%rbp), %eax
	andl	$63, %eax
	movl	$1, %edi
	movl	%eax, %ecx
	salq	%cl, %rdi
	movq	%rdi, %rax
	notq	%rax
	andq	%rax, %rdx
	movslq	%esi, %rax
	movq	%rdx, -1680(%rbp,%rax,8)
	movl	$0, -1848(%rbp)
	movq	$48, -1800(%rbp)
	jmp	.L111
.L103:
	leaq	.LC9(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movq	$69, -1800(%rbp)
	jmp	.L111
.L104:
	leaq	.LC10(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	-1860(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	-1860(%rbp), %eax
	leal	63(%rax), %edx
	testl	%eax, %eax
	cmovs	%edx, %eax
	sarl	$6, %eax
	movl	%eax, %esi
	movslq	%esi, %rax
	movq	-1680(%rbp,%rax,8), %rdx
	movl	-1860(%rbp), %eax
	andl	$63, %eax
	movl	$1, %edi
	movl	%eax, %ecx
	salq	%cl, %rdi
	movq	%rdi, %rax
	notq	%rax
	andq	%rax, %rdx
	movslq	%esi, %rax
	movq	%rdx, -1680(%rbp,%rax,8)
	movq	$69, -1800(%rbp)
	jmp	.L111
.L98:
	leaq	-1040(%rbp), %rsi
	movl	-1860(%rbp), %eax
	movl	$0, %ecx
	movl	$1024, %edx
	movl	%eax, %edi
	call	recv@PLT
	movq	%rax, -1768(%rbp)
	movq	-1768(%rbp), %rax
	movq	%rax, -1816(%rbp)
	movq	$24, -1800(%rbp)
	jmp	.L111
.L92:
	cmpq	$0, -1816(%rbp)
	jne	.L130
	movq	$45, -1800(%rbp)
	jmp	.L111
.L130:
	movq	$9, -1800(%rbp)
	jmp	.L111
.L100:
	leaq	-1040(%rbp), %rdx
	movq	-1816(%rbp), %rax
	addq	%rdx, %rax
	movb	$0, (%rax)
	leaq	-1040(%rbp), %rdx
	movl	-1860(%rbp), %eax
	movl	%eax, %esi
	leaq	.LC11(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	leaq	-1040(%rbp), %rax
	movl	$4, %edx
	leaq	.LC12(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	call	strncmp@PLT
	movl	%eax, -1840(%rbp)
	movq	$23, -1800(%rbp)
	jmp	.L111
.L57:
	cmpl	$9, -1852(%rbp)
	jg	.L132
	movq	$33, -1800(%rbp)
	jmp	.L111
.L132:
	movq	$69, -1800(%rbp)
	jmp	.L111
.L85:
	cmpl	$-1, -1864(%rbp)
	jne	.L134
	movq	$61, -1800(%rbp)
	jmp	.L111
.L134:
	movq	$59, -1800(%rbp)
	jmp	.L111
.L72:
	movl	$0, -1860(%rbp)
	movq	$36, -1800(%rbp)
	jmp	.L111
.L107:
	cmpl	$9, -1836(%rbp)
	jg	.L136
	movq	$53, -1800(%rbp)
	jmp	.L111
.L136:
	movq	$69, -1800(%rbp)
	jmp	.L111
.L87:
	movl	-1852(%rbp), %eax
	cltq
	movl	-1856(%rbp), %edx
	movl	%edx, -1728(%rbp,%rax,4)
	movl	-1856(%rbp), %eax
	leal	63(%rax), %edx
	testl	%eax, %eax
	cmovs	%edx, %eax
	sarl	$6, %eax
	movl	%eax, %esi
	movslq	%esi, %rax
	movq	-1680(%rbp,%rax,8), %rdx
	movl	-1856(%rbp), %eax
	andl	$63, %eax
	movl	$1, %edi
	movl	%eax, %ecx
	salq	%cl, %rdi
	movq	%rdi, %rax
	orq	%rax, %rdx
	movslq	%esi, %rax
	movq	%rdx, -1680(%rbp,%rax,8)
	movq	$69, -1800(%rbp)
	jmp	.L111
.L71:
	leaq	.LC13(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	-1884(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	$1, %edi
	call	exit@PLT
.L73:
	movq	-1680(%rbp), %rax
	movq	-1672(%rbp), %rdx
	movq	%rax, -1552(%rbp)
	movq	%rdx, -1544(%rbp)
	movq	-1664(%rbp), %rax
	movq	-1656(%rbp), %rdx
	movq	%rax, -1536(%rbp)
	movq	%rdx, -1528(%rbp)
	movq	-1648(%rbp), %rax
	movq	-1640(%rbp), %rdx
	movq	%rax, -1520(%rbp)
	movq	%rdx, -1512(%rbp)
	movq	-1632(%rbp), %rax
	movq	-1624(%rbp), %rdx
	movq	%rax, -1504(%rbp)
	movq	%rdx, -1496(%rbp)
	movq	-1616(%rbp), %rax
	movq	-1608(%rbp), %rdx
	movq	%rax, -1488(%rbp)
	movq	%rdx, -1480(%rbp)
	movq	-1600(%rbp), %rax
	movq	-1592(%rbp), %rdx
	movq	%rax, -1472(%rbp)
	movq	%rdx, -1464(%rbp)
	movq	-1584(%rbp), %rax
	movq	-1576(%rbp), %rdx
	movq	%rax, -1456(%rbp)
	movq	%rdx, -1448(%rbp)
	movq	-1568(%rbp), %rax
	movq	-1560(%rbp), %rdx
	movq	%rax, -1440(%rbp)
	movq	%rdx, -1432(%rbp)
	leaq	-1552(%rbp), %rax
	movl	$0, %r8d
	movl	$0, %ecx
	movl	$0, %edx
	movq	%rax, %rsi
	movl	$1024, %edi
	call	select@PLT
	movl	%eax, -1864(%rbp)
	movq	$40, -1800(%rbp)
	jmp	.L111
.L64:
	leaq	-1040(%rbp), %rax
	leaq	.LC6(%rip), %rdx
	movq	%rdx, %rsi
	movq	%rax, %rdi
	call	strtok@PLT
	movq	%rax, -1792(%rbp)
	movq	-1792(%rbp), %rax
	movq	%rax, -1808(%rbp)
	leaq	.LC6(%rip), %rax
	movq	%rax, %rsi
	movl	$0, %edi
	call	strtok@PLT
	movq	%rax, -1808(%rbp)
	movq	$18, -1800(%rbp)
	jmp	.L111
.L63:
	movl	-1836(%rbp), %eax
	cltq
	movl	$-1, -1728(%rbp,%rax,4)
	movq	$69, -1800(%rbp)
	jmp	.L111
.L78:
	cmpl	$9, -1848(%rbp)
	jg	.L138
	movq	$42, -1800(%rbp)
	jmp	.L111
.L138:
	movq	$69, -1800(%rbp)
	jmp	.L111
.L93:
	cmpl	$-1, -1884(%rbp)
	jne	.L140
	movq	$77, -1800(%rbp)
	jmp	.L111
.L140:
	movq	$70, -1800(%rbp)
	jmp	.L111
.L76:
	movl	-1836(%rbp), %eax
	cltq
	movl	-1728(%rbp,%rax,4), %eax
	cmpl	%eax, -1860(%rbp)
	jne	.L142
	movq	$75, -1800(%rbp)
	jmp	.L111
.L142:
	movq	$49, -1800(%rbp)
	jmp	.L111
.L79:
	movq	-1824(%rbp), %rax
	movl	-1872(%rbp), %edx
	movq	$0, (%rax,%rdx,8)
	addl	$1, -1872(%rbp)
	movq	$92, -1800(%rbp)
	jmp	.L111
.L65:
	cmpl	$-1, -1856(%rbp)
	je	.L144
	movq	$37, -1800(%rbp)
	jmp	.L111
.L144:
	movq	$69, -1800(%rbp)
	jmp	.L111
.L56:
	leaq	-1040(%rbp), %rcx
	leaq	-1680(%rbp), %rdx
	leaq	-1728(%rbp), %rsi
	movl	-1860(%rbp), %eax
	movl	%eax, %edi
	call	broadcast
	movq	$69, -1800(%rbp)
	jmp	.L111
.L91:
	movl	-1852(%rbp), %eax
	cltq
	movl	-1728(%rbp,%rax,4), %eax
	cmpl	$-1, %eax
	jne	.L146
	movq	$38, -1800(%rbp)
	jmp	.L111
.L146:
	movq	$83, -1800(%rbp)
	jmp	.L111
.L88:
	leaq	.LC14(%rip), %rax
	movq	%rax, %rdi
	call	puts@PLT
	movl	userCounter(%rip), %edx
	leaq	-1424(%rbp), %rax
	movl	%edx, %esi
	movq	%rax, %rdi
	call	generateDefaultName
	leaq	-1424(%rbp), %rdx
	leaq	-1296(%rbp), %rax
	movq	%rdx, %rcx
	leaq	.LC15(%rip), %rdx
	movl	$128, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-1296(%rbp), %rcx
	leaq	-1680(%rbp), %rdx
	leaq	-1728(%rbp), %rsi
	movl	-1856(%rbp), %eax
	movl	%eax, %edi
	call	broadcast
	leaq	-1424(%rbp), %rdx
	movl	-1856(%rbp), %eax
	movq	%rdx, %rsi
	movl	%eax, %edi
	call	sendWelcomeMessage
	movl	$0, -1852(%rbp)
	movq	$90, -1800(%rbp)
	jmp	.L111
.L69:
	addl	$1, -1848(%rbp)
	movq	$48, -1800(%rbp)
	jmp	.L111
.L84:
	leaq	-1888(%rbp), %rdx
	leaq	-1744(%rbp), %rcx
	movl	-1884(%rbp), %eax
	movq	%rcx, %rsi
	movl	%eax, %edi
	call	accept@PLT
	movl	%eax, -1828(%rbp)
	movl	-1828(%rbp), %eax
	movl	%eax, -1856(%rbp)
	movq	$73, -1800(%rbp)
	jmp	.L111
.L52:
	movq	-1904(%rbp), %rax
	movq	(%rax), %rdx
	movq	stderr(%rip), %rax
	leaq	.LC16(%rip), %rcx
	movq	%rcx, %rsi
	movq	%rax, %rdi
	movl	$0, %eax
	call	fprintf@PLT
	movl	$1, %edi
	call	exit@PLT
.L55:
	cmpl	$15, -1872(%rbp)
	ja	.L148
	movq	$47, -1800(%rbp)
	jmp	.L111
.L148:
	movq	$35, -1800(%rbp)
	jmp	.L111
.L83:
	movl	-1848(%rbp), %eax
	cltq
	movl	-1728(%rbp,%rax,4), %eax
	cmpl	%eax, -1860(%rbp)
	jne	.L150
	movq	$8, -1800(%rbp)
	jmp	.L111
.L150:
	movq	$64, -1800(%rbp)
	jmp	.L111
.L80:
	cmpl	$-1, -1880(%rbp)
	jne	.L152
	movq	$43, -1800(%rbp)
	jmp	.L111
.L152:
	movq	$12, -1800(%rbp)
	jmp	.L111
.L86:
	movq	-1904(%rbp), %rax
	addq	$8, %rax
	movq	(%rax), %rax
	movq	%rax, %rsi
	leaq	.LC17(%rip), %rax
	movq	%rax, %rdi
	movl	$0, %eax
	call	printf@PLT
	movq	$21, -1800(%rbp)
	jmp	.L111
.L59:
	addl	$1, -1852(%rbp)
	movq	$90, -1800(%rbp)
	jmp	.L111
.L106:
	movl	$16, -1888(%rbp)
	movl	$0, %edx
	movl	$1, %esi
	movl	$2, %edi
	call	socket@PLT
	movl	%eax, -1884(%rbp)
	movq	$28, -1800(%rbp)
	jmp	.L111
.L90:
	movl	-1884(%rbp), %eax
	leal	63(%rax), %edx
	testl	%eax, %eax
	cmovs	%edx, %eax
	sarl	$6, %eax
	movl	%eax, %esi
	movslq	%esi, %rax
	movq	-1680(%rbp,%rax,8), %rdx
	movl	-1884(%rbp), %eax
	andl	$63, %eax
	movl	$1, %edi
	movl	%eax, %ecx
	salq	%cl, %rdi
	movq	%rdi, %rax
	orq	%rax, %rdx
	movslq	%esi, %rax
	movq	%rdx, -1680(%rbp,%rax,8)
	movl	$0, -1868(%rbp)
	movq	$68, -1800(%rbp)
	jmp	.L111
.L82:
	leaq	.LC18(%rip), %rax
	movq	%rax, %rdi
	call	perror@PLT
	movl	-1884(%rbp), %eax
	movl	%eax, %edi
	call	close@PLT
	movl	$1, %edi
	call	exit@PLT
.L156:
	nop
.L111:
	jmp	.L154
	.cfi_endproc
.LFE9:
	.size	main, .-main
	.section	.rodata
.LC19:
	.string	"Welcome, %s!"
	.text
	.globl	sendWelcomeMessage
	.type	sendWelcomeMessage, @function
sendWelcomeMessage:
.LFB10:
	.cfi_startproc
	endbr64
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset 6, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register 6
	subq	$176, %rsp
	movl	%edi, -164(%rbp)
	movq	%rsi, -176(%rbp)
	movq	%fs:40, %rax
	movq	%rax, -8(%rbp)
	xorl	%eax, %eax
	movq	$1, -160(%rbp)
.L163:
	cmpq	$2, -160(%rbp)
	je	.L158
	cmpq	$2, -160(%rbp)
	ja	.L166
	cmpq	$0, -160(%rbp)
	je	.L167
	cmpq	$1, -160(%rbp)
	jne	.L166
	movq	$2, -160(%rbp)
	jmp	.L161
.L158:
	movq	-176(%rbp), %rdx
	leaq	-144(%rbp), %rax
	movq	%rdx, %rcx
	leaq	.LC19(%rip), %rdx
	movl	$128, %esi
	movq	%rax, %rdi
	movl	$0, %eax
	call	snprintf@PLT
	leaq	-144(%rbp), %rax
	movq	%rax, %rdi
	call	strlen@PLT
	movq	%rax, -152(%rbp)
	movq	-152(%rbp), %rdx
	leaq	-144(%rbp), %rsi
	movl	-164(%rbp), %eax
	movl	$0, %ecx
	movl	%eax, %edi
	call	send@PLT
	movq	$0, -160(%rbp)
	jmp	.L161
.L166:
	nop
.L161:
	jmp	.L163
.L167:
	nop
	movq	-8(%rbp), %rax
	subq	%fs:40, %rax
	je	.L165
	call	__stack_chk_fail@PLT
.L165:
	leave
	.cfi_def_cfa 7, 8
	ret
	.cfi_endproc
.LFE10:
	.size	sendWelcomeMessage, .-sendWelcomeMessage
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
